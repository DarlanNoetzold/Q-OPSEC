"""Validação estática dos manifests k8s do Q-OPSEC (KB-005).

Não substitui `kubectl apply` num cluster, mas pega as quebras que impedem a
aplicação: seletor que não casa com os labels do pod, Service apontando para
porta inexistente, HPA/PDB/Ingress referenciando objeto ausente, container sem
limits ou imagem sem tag fixa.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
K8S_DIR = ROOT / "deploy/k8s"
NAMESPACE = "qopsec"

CLUSTER_SCOPED = {"Namespace", "ClusterRole", "ClusterRoleBinding", "APIService"}
WORKLOADS = {"Deployment", "StatefulSet"}


def load_docs():
    docs = []
    for path in sorted(K8S_DIR.glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc:
                docs.append((path.name, doc))
    return docs


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader que recusa chaves duplicadas no mesmo mapeamento."""


def _construct_mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"chave duplicada: {key}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


@pytest.mark.parametrize("path", sorted(K8S_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_manifest_has_no_duplicate_keys(path):
    try:
        list(yaml.load_all(path.read_text(encoding="utf-8"), Loader=_StrictLoader))
    except yaml.constructor.ConstructorError as exc:
        pytest.fail(f"{path.name}: {exc}")


DOCS = load_docs()


def kind_of(doc):
    return doc.get("kind")


def name_of(doc):
    return doc.get("metadata", {}).get("name")


def ns_of(doc):
    return doc.get("metadata", {}).get("namespace")


def by_kind(*kinds):
    return [(f, d) for f, d in DOCS if kind_of(d) in kinds]


def test_manifests_exist():
    assert K8S_DIR.is_dir(), "diretório deploy/k8s ausente"
    assert len(DOCS) >= 20, f"esperava os manifests completos, achei {len(DOCS)} documentos"


@pytest.mark.parametrize("filename,doc", DOCS, ids=[f"{f}:{kind_of(d)}:{name_of(d)}" for f, d in DOCS])
def test_every_document_has_identity(filename, doc):
    assert doc.get("apiVersion"), f"{filename}: apiVersion ausente"
    assert kind_of(doc), f"{filename}: kind ausente"
    assert name_of(doc), f"{filename}: metadata.name ausente"


@pytest.mark.parametrize("filename,doc", DOCS, ids=[f"{f}:{kind_of(d)}:{name_of(d)}" for f, d in DOCS])
def test_namespaced_objects_live_in_qopsec(filename, doc):
    if kind_of(doc) in CLUSTER_SCOPED:
        return
    namespace = ns_of(doc)
    if namespace == "kube-system":
        # O adapter precisa ler extension-apiserver-authentication no kube-system.
        assert kind_of(doc) == "RoleBinding", f"{filename}: só RoleBinding pode ir para kube-system"
        return
    assert namespace == NAMESPACE, f"{filename}: {kind_of(doc)}/{name_of(doc)} fora do namespace {NAMESPACE}"


def test_no_duplicate_objects():
    seen = {}
    for filename, doc in DOCS:
        key = (kind_of(doc), ns_of(doc), name_of(doc))
        assert key not in seen, f"{kind_of(doc)}/{name_of(doc)} duplicado em {filename} e {seen[key]}"
        seen[key] = filename


@pytest.mark.parametrize(
    "filename,doc", by_kind(*WORKLOADS), ids=[f"{f}:{name_of(d)}" for f, d in by_kind(*WORKLOADS)]
)
def test_workload_selector_matches_pod_labels(filename, doc):
    spec = doc["spec"]
    selector = spec.get("selector", {}).get("matchLabels", {})
    labels = spec["template"]["metadata"].get("labels", {})
    assert selector, f"{filename}: {name_of(doc)} sem selector.matchLabels"
    missing = {k: v for k, v in selector.items() if labels.get(k) != v}
    assert not missing, f"{filename}: labels do pod não casam com o selector: {missing}"


@pytest.mark.parametrize(
    "filename,doc", by_kind(*WORKLOADS), ids=[f"{f}:{name_of(d)}" for f, d in by_kind(*WORKLOADS)]
)
def test_containers_have_limits_and_pinned_images(filename, doc):
    for container in doc["spec"]["template"]["spec"]["containers"]:
        name = container["name"]
        image = container["image"]
        assert ":" in image, f"{filename}: container {name} sem tag explícita ({image})"
        assert not image.endswith(":latest"), f"{filename}: container {name} usa :latest ({image})"
        resources = container.get("resources", {})
        assert resources.get("requests"), f"{filename}: container {name} sem resources.requests"
        assert resources.get("limits"), f"{filename}: container {name} sem resources.limits"


@pytest.mark.parametrize(
    "filename,doc", by_kind(*WORKLOADS), ids=[f"{f}:{name_of(d)}" for f, d in by_kind(*WORKLOADS)]
)
def test_containers_declare_probes(filename, doc):
    for container in doc["spec"]["template"]["spec"]["containers"]:
        assert container.get("readinessProbe"), f"{filename}: container {container['name']} sem readinessProbe"
        assert container.get("livenessProbe"), f"{filename}: container {container['name']} sem livenessProbe"


def pod_labels_of(doc):
    return doc["spec"]["template"]["metadata"].get("labels", {})


def container_ports_of(doc):
    ports = {}
    for container in doc["spec"]["template"]["spec"]["containers"]:
        for port in container.get("ports", []) or []:
            if port.get("name"):
                ports[port["name"]] = port.get("containerPort")
    return ports


@pytest.mark.parametrize("filename,doc", by_kind("Service"), ids=[f"{f}:{name_of(d)}" for f, d in by_kind("Service")])
def test_service_selects_an_existing_workload(filename, doc):
    selector = doc["spec"].get("selector", {})
    assert selector, f"{filename}: Service {name_of(doc)} sem selector"
    matches = [
        (f, d)
        for f, d in by_kind(*WORKLOADS)
        if ns_of(d) == ns_of(doc) and all(pod_labels_of(d).get(k) == v for k, v in selector.items())
    ]
    assert matches, f"{filename}: Service {name_of(doc)} não seleciona nenhum pod ({selector})"


@pytest.mark.parametrize("filename,doc", by_kind("Service"), ids=[f"{f}:{name_of(d)}" for f, d in by_kind("Service")])
def test_service_target_port_resolves(filename, doc):
    selector = doc["spec"].get("selector", {})
    targets = [
        d
        for _, d in by_kind(*WORKLOADS)
        if ns_of(d) == ns_of(doc) and all(pod_labels_of(d).get(k) == v for k, v in selector.items())
    ]
    if not targets:
        pytest.skip("sem workload correspondente; coberto pelo teste de selector")
    ports = container_ports_of(targets[0])
    for port in doc["spec"].get("ports", []):
        target = port.get("targetPort", port.get("port"))
        if isinstance(target, str):
            assert target in ports, f"{filename}: targetPort '{target}' não existe no pod ({sorted(ports)})"
        else:
            assert target in ports.values(), f"{filename}: targetPort {target} não existe no pod ({sorted(ports.values())})"


@pytest.mark.parametrize(
    "filename,doc", by_kind("HorizontalPodAutoscaler"), ids=[f"{f}:{name_of(d)}" for f, d in by_kind("HorizontalPodAutoscaler")]
)
def test_hpa_targets_existing_workload(filename, doc):
    ref = doc["spec"]["scaleTargetRef"]
    found = [
        d
        for _, d in by_kind(*WORKLOADS)
        if kind_of(d) == ref["kind"] and name_of(d) == ref["name"] and ns_of(d) == ns_of(doc)
    ]
    assert found, f"{filename}: HPA aponta para {ref['kind']}/{ref['name']} inexistente"
    metrics = doc["spec"].get("metrics", [])
    assert metrics, f"{filename}: HPA sem metrics"
    for metric in metrics:
        assert metric.get("type") in {"Resource", "External", "Pods", "Object"}, f"{filename}: tipo inválido {metric}"
        payload = metric.get(metric["type"].lower(), {})
        assert payload.get("name") or payload.get("metric", {}).get("name"), f"{filename}: métrica sem nome"


def test_at_least_one_hpa_scales_on_queue_depth():
    external = []
    for filename, doc in by_kind("HorizontalPodAutoscaler"):
        for metric in doc["spec"].get("metrics", []):
            if metric.get("type") == "External":
                external.append((filename, metric["external"]["metric"]["name"]))
    assert external, "nenhum HPA escala por métrica External (profundidade de fila)"
    assert any("queue" in name or "depth" in name for _, name in external), f"métrica external sem relação com fila: {external}"


@pytest.mark.parametrize(
    "filename,doc", by_kind("PodDisruptionBudget"), ids=[f"{f}:{name_of(d)}" for f, d in by_kind("PodDisruptionBudget")]
)
def test_pdb_selects_an_existing_workload(filename, doc):
    selector = doc["spec"].get("selector", {}).get("matchLabels", {})
    matches = [
        d
        for _, d in by_kind(*WORKLOADS)
        if ns_of(d) == ns_of(doc) and all(pod_labels_of(d).get(k) == v for k, v in selector.items())
    ]
    assert matches, f"{filename}: PDB {name_of(doc)} não seleciona nenhum pod ({selector})"


def test_ingress_backends_exist():
    services = {name_of(d) for _, d in by_kind("Service")}
    for filename, doc in by_kind("Ingress"):
        for rule in doc["spec"].get("rules", []):
            for path in rule.get("http", {}).get("paths", []):
                backend = path["backend"]["service"]["name"]
                assert backend in services, f"{filename}: Ingress aponta para Service {backend} inexistente"


def test_prometheus_adapter_serves_external_metrics_apiservice():
    apiservices = [d for _, d in by_kind("APIService")]
    assert apiservices, "APIService do prometheus-adapter ausente"
    names = {name_of(d) for d in apiservices}
    assert "v1beta2.external.metrics.k8s.io" in names, f"APIService external ausente: {sorted(names)}"
    for doc in apiservices:
        assert doc["spec"]["service"]["name"] == "prometheus-adapter"
        assert doc["spec"]["service"]["namespace"] == NAMESPACE


def test_prometheus_adapter_has_rbac_for_apiserver_auth():
    bindings = {(kind_of(d), name_of(d)): d for _, d in by_kind("ClusterRoleBinding", "RoleBinding")}
    delegator = bindings.get(("ClusterRoleBinding", "qopsec-prometheus-adapter-auth-delegator"))
    assert delegator, "falta ClusterRoleBinding para system:auth-delegator"
    assert delegator["roleRef"]["name"] == "system:auth-delegator"
    reader = bindings.get(("RoleBinding", "qopsec-prometheus-adapter-auth-reader"))
    assert reader, "falta RoleBinding para extension-apiserver-authentication-reader"
    assert reader["metadata"]["namespace"] == "kube-system"
    assert reader["roleRef"]["name"] == "extension-apiserver-authentication-reader"


def test_prometheus_scrapes_the_workloads():
    configs = [d for _, d in by_kind("ConfigMap") if name_of(d) == "prometheus-config"]
    assert configs, "ConfigMap prometheus-config ausente"
    scrape = configs[0]["data"]["prometheus.yml"]
    assert "kubernetes_sd_configs" in scrape
    assert "qopsec-(orchestrator|kms)" in scrape, "relabel não restringe os pods do Q-OPSEC"
    assert "container_port_name" in scrape or "container_port" in scrape
