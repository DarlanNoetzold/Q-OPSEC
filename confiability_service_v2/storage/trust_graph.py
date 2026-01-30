"""
Trust Graph - Grafo de relacionamentos de confiança entre entidades
"""
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import threading


class TrustNode:
    """Nó no grafo de confiança"""

    def __init__(self, node_id: str, node_type: str):
        self.node_id = node_id
        self.node_type = node_type  # "source", "entity", "datatype", "author"
        self.trust_scores: List[float] = []
        self.metadata: Dict = {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    @property
    def avg_trust(self) -> float:
        """Retorna trust score médio do nó"""
        return sum(self.trust_scores) / len(self.trust_scores) if self.trust_scores else 0.5

    def add_trust_score(self, score: float):
        """Adiciona um novo trust score"""
        self.trust_scores.append(score)
        self.updated_at = datetime.utcnow()

        # Manter apenas últimos 100 scores
        if len(self.trust_scores) > 100:
            self.trust_scores = self.trust_scores[-100:]


class TrustEdge:
    """Aresta no grafo de confiança (relacionamento entre nós)"""

    def __init__(
            self,
            from_node: str,
            to_node: str,
            edge_type: str,
            weight: float = 1.0
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.edge_type = edge_type  # "produces", "references", "contradicts", "confirms"
        self.weight = weight
        self.interactions = 0
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def strengthen(self, amount: float = 0.1):
        """Fortalece a aresta"""
        self.weight = min(1.0, self.weight + amount)
        self.interactions += 1
        self.updated_at = datetime.utcnow()

    def weaken(self, amount: float = 0.1):
        """Enfraquece a aresta"""
        self.weight = max(0.0, self.weight - amount)
        self.interactions += 1
        self.updated_at = datetime.utcnow()


class TrustGraph:
    """
    Grafo de confiança para modelar relacionamentos entre:
    - Sources (fontes de informação)
    - Entities (entidades/claims)
    - Authors (autores)
    - DataTypes (tipos de dados)

    Permite análise de:
    - Propagação de confiança
    - Detecção de clusters suspeitos
    - Cross-validation entre fontes
    - Identificação de fontes correlacionadas
    """

    def __init__(self):
        self._nodes: Dict[str, TrustNode] = {}
        self._edges: Dict[Tuple[str, str], TrustEdge] = {}

        # Índices para busca rápida
        self._outgoing: Dict[str, Set[str]] = defaultdict(set)  # node -> neighbors
        self._incoming: Dict[str, Set[str]] = defaultdict(set)  # node -> sources

        self._lock = threading.Lock()

    def add_node(
            self,
            node_id: str,
            node_type: str,
            metadata: Optional[Dict] = None
    ) -> TrustNode:
        """
        Adiciona ou atualiza um nó no grafo

        Args:
            node_id: ID único do nó
            node_type: Tipo do nó (source, entity, author, datatype)
            metadata: Metadados adicionais

        Returns:
            TrustNode criado ou existente
        """
        with self._lock:
            if node_id not in self._nodes:
                node = TrustNode(node_id, node_type)
                if metadata:
                    node.metadata = metadata
                self._nodes[node_id] = node
            else:
                node = self._nodes[node_id]
                if metadata:
                    node.metadata.update(metadata)

            return node

    def add_edge(
            self,
            from_node: str,
            to_node: str,
            edge_type: str,
            weight: float = 1.0
    ) -> TrustEdge:
        """
        Adiciona ou atualiza uma aresta no grafo

        Args:
            from_node: ID do nó de origem
            to_node: ID do nó de destino
            edge_type: Tipo de relacionamento
            weight: Peso da aresta (0-1)

        Returns:
            TrustEdge criada ou existente
        """
        with self._lock:
            edge_key = (from_node, to_node)

            if edge_key not in self._edges:
                edge = TrustEdge(from_node, to_node, edge_type, weight)
                self._edges[edge_key] = edge
                self._outgoing[from_node].add(to_node)
                self._incoming[to_node].add(from_node)
            else:
                edge = self._edges[edge_key]
                edge.strengthen(0.05)

            return edge

    def get_node(self, node_id: str) -> Optional[TrustNode]:
        """Retorna um nó pelo ID"""
        return self._nodes.get(node_id)

    def get_neighbors(
            self,
            node_id: str,
            direction: str = "outgoing"
    ) -> List[TrustNode]:
        """
        Retorna vizinhos de um nó

        Args:
            node_id: ID do nó
            direction: "outgoing" (saindo) ou "incoming" (chegando)

        Returns:
            Lista de nós vizinhos
        """
        if direction == "outgoing":
            neighbor_ids = self._outgoing.get(node_id, set())
        else:
            neighbor_ids = self._incoming.get(node_id, set())

        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def calculate_node_trust(
            self,
            node_id: str,
            depth: int = 2
    ) -> float:
        """
        Calcula trust score de um nó baseado em seus vizinhos (propagação)

        Args:
            node_id: ID do nó
            depth: Profundidade da propagação

        Returns:
            Trust score calculado (0-1)
        """
        node = self.get_node(node_id)
        if not node:
            return 0.5

        # Trust direto do nó
        direct_trust = node.avg_trust

        if depth == 0:
            return direct_trust

        # Trust propagado dos vizinhos
        neighbors = self.get_neighbors(node_id, "incoming")

        if not neighbors:
            return direct_trust

        neighbor_trusts = []
        for neighbor in neighbors:
            edge_key = (neighbor.node_id, node_id)
            edge = self._edges.get(edge_key)

            if edge:
                neighbor_trust = self.calculate_node_trust(
                    neighbor.node_id,
                    depth - 1
                )
                weighted_trust = neighbor_trust * edge.weight
                neighbor_trusts.append(weighted_trust)

        if not neighbor_trusts:
            return direct_trust

        propagated_trust = sum(neighbor_trusts) / len(neighbor_trusts)

        # Combinar trust direto (70%) com propagado (30%)
        return (direct_trust * 0.7) + (propagated_trust * 0.3)

    def find_clusters(
            self,
            min_cluster_size: int = 3
    ) -> List[Set[str]]:
        """
        Identifica clusters de nós fortemente conectados

        Args:
            min_cluster_size: Tamanho mínimo do cluster

        Returns:
            Lista de clusters (conjuntos de node_ids)
        """
        visited = set()
        clusters = []

        def dfs(node_id: str, cluster: Set[str]):
            if node_id in visited:
                return

            visited.add(node_id)
            cluster.add(node_id)

            # Explorar vizinhos com peso > 0.5
            for neighbor_id in self._outgoing.get(node_id, set()):
                edge = self._edges.get((node_id, neighbor_id))
                if edge and edge.weight > 0.5:
                    dfs(neighbor_id, cluster)

        for node_id in self._nodes:
            if node_id not in visited:
                cluster = set()
                dfs(node_id, cluster)

                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

        return clusters

    def detect_suspicious_patterns(self) -> List[Dict]:
        """
        Detecta padrões suspeitos no grafo

        Returns:
            Lista de padrões suspeitos detectados
        """
        patterns = []

        # Padrão 1: Nós com muitas conexões de baixa confiança
        for node_id, node in self._nodes.items():
            if node.avg_trust < 0.3:
                out_degree = len(self._outgoing.get(node_id, set()))
                if out_degree > 10:
                    patterns.append({
                        "type": "low_trust_hub",
                        "node_id": node_id,
                        "node_type": node.node_type,
                        "trust": node.avg_trust,
                        "connections": out_degree
                    })

        # Padrão 2: Clusters isolados de baixa confiança
        clusters = self.find_clusters(min_cluster_size=3)
        for cluster in clusters:
            avg_trust = sum(
                self._nodes[nid].avg_trust
                for nid in cluster
                if nid in self._nodes
            ) / len(cluster)

            if avg_trust < 0.4:
                patterns.append({
                    "type": "low_trust_cluster",
                    "cluster_size": len(cluster),
                    "avg_trust": avg_trust,
                    "nodes": list(cluster)
                })

        return patterns

    def get_cross_validation_score(
            self,
            entity_id: str,
            min_sources: int = 2
    ) -> float:
        """
        Calcula score de validação cruzada para uma entidade

        Args:
            entity_id: ID da entidade
            min_sources: Mínimo de fontes para validação

        Returns:
            Score de validação cruzada (0-1)
        """
        # Encontrar sources que referenciam esta entidade
        sources = [
            node for node in self.get_neighbors(entity_id, "incoming")
            if node.node_type == "source"
        ]

        if len(sources) < min_sources:
            return 0.3  # Baixa validação

        # Calcular trust médio das sources
        avg_source_trust = sum(s.avg_trust for s in sources) / len(sources)

        # Bonus por múltiplas sources
        diversity_bonus = min(0.2, len(sources) * 0.05)

        return min(1.0, avg_source_trust + diversity_bonus)

    def clear(self):
        """Limpa o grafo (útil para testes)"""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._outgoing.clear()
            self._incoming.clear()