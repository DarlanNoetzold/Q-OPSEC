#!/usr/bin/env python3
import asyncio
import aiohttp
import random
import argparse
import json
import time
import uuid
import hashlib
from typing import Dict, Any, Tuple, List

# ---------- Catálogos básicos ----------

DOC_TYPES = ["finance", "hr", "legal", "marketing", "support", "engineering", "sales", "ops", "security", "product", "compliance"]
APPS = ["mobile", "web", "admin", "partner", "api", "cli", "edge", "iot"]
DESTS = [
    "service-payments", "service-hr", "service-legal", "service-analytics",
    "service-support", "service-risk", "service-dlp", "service-kde", "service-kms"
]
SEVERITIES = ["low", "medium", "high", "critical"]

LANGS = ["en", "pt-BR", "es", "fr", "de", "ja"]
LOCALES = ["en_US", "pt_BR", "es_MX", "fr_FR", "de_DE", "ja_JP"]

USER_LABELS = ["public", "internal", "confidential", "restricted"]
REGIONS = ["us", "eu", "latam", "apac", "me"]

PASSWORDS = ["admin123", "P@ssw0rd", "Secret_2025!", "letmein", "Tr0ub4dor&3"]
SECRETS = ["sk_live_9f8a7s6d5f4", "AKIAIOSFODNN7EXAMPLE", "ghp_abc123xyz987", "l1v3-prod-KEY"]
CARDS = ["4111111111111111", "5555555555554444", "4000000000000002", "378282246310005"]  # testes
PIX_KEYS = ["cpf:12345678901", "cnpj:12345678000199", "email:pagamentos@empresa.com", "phone:+5511999999999", "aleatoria:1f2e3d4c-5b6a-7f8e-9a0b-1c2d3e4f5a6b"]

EMAIL_USERS = ["john.doe", "jane.smith", "admin", "finance.bot", "alerts", "maria.silva", "joao.souza"]
EMAIL_DOMAINS = ["example.com", "corp.local", "partner.net", "company.io", "org.br", "empresa.com.br"]

OS_LIST = ["ios", "android", "windows", "linux", "macos"]
DEVICES = ["phone", "tablet", "desktop", "server", "gateway", "sensor"]
NETWORKS = ["wifi", "lte", "ethernet", "5g", "satellite"]

ALGOS = [
    {"name": "AES-GCM-256", "type": "classical"},
    {"name": "CHACHA20-POLY1305", "type": "classical"},
    {"name": "Kyber-768", "type": "pqc"},
    {"name": "Dilithium-2", "type": "pqc"},
    {"name": "QKD", "type": "qkd"},
]
DELIVERY_HINTS = ["api", "mqtt", "file", "hsm"]

# ---------- Templates de mensagens ----------

PUBLIC_TEMPLATES_EN = [
    "Product update announcement for Q{q} {year}.",
    "Open API call for feedback on new feature beta.",
    "General notice to all users regarding terms update."
]
PUBLIC_TEMPLATES_PT = [
    "Comunicado de atualização de produto para o Q{q} de {year}.",
    "Chamada aberta para feedback da nova funcionalidade beta.",
    "Aviso geral aos usuários sobre atualização de termos."
]

INTERNAL_TEMPLATES = [
    "Internal memo about sprint {sprint} retrospective and roadmap items.",
    "Meeting notes for project Phoenix: phase {phase} planning.",
    "Team-only discussion on process improvements and KPIs."
]

CONFIDENTIAL_TEMPLATES = [
    "Client {cid} contract negotiation includes pricing tiers and discounts.",
    "HR review for employee {eid}: performance band and salary range.",
    "Customer PII includes email {email} and phone +1-{p1}-{p2}-{p3}."
]

RESTRICTED_TEMPLATES = [
    "Transfer request for client {cid}, card {card}",
    "Rotate admin credentials, current password {pwd} expires today.",
    "Deploy secret key {secret} to production vault immediately.",
    "PIX transfer using key {pix} scheduled for tomorrow."
]

NOISE_TEMPLATES = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "随机内容混合 English with unicode — café, naïve, résumé.",
    "RTL test: ‏مرحبا بالعالم – data integrity ✓",
    "SQL test: SELECT * FROM users WHERE email='{email}' OR '1'='1'; --",
    "JSON-like: {{ 'test': true, 'count': {count} }}",
]

# ---------- Helpers ----------

def rand_email(rng: random.Random) -> str:
    return f"{rng.choice(EMAIL_USERS)}@{rng.choice(EMAIL_DOMAINS)}"

def rand_phone(rng: random.Random) -> Tuple[int, int, int]:
    return rng.randint(201, 989), rng.randint(100, 999), rng.randint(1000, 9999)

def rand_locale(rng: random.Random) -> Tuple[str, str]:
    return rng.choice(LANGS), rng.choice(LOCALES)

def rand_user(rng: random.Random) -> Dict[str, Any]:
    return {
        "user_id": f"u-{rng.randint(1000,999999)}",
        "email": rand_email(rng),
        "role": rng.choice(["user", "admin", "ops", "auditor", "service"]),
        "tenant_id": f"t-{rng.randint(10,999)}" if rng.random() < 0.7 else None
    }

def rand_device_info(rng: random.Random) -> Dict[str, Any]:
    return {
        "os": rng.choice(OS_LIST),
        "device_type": rng.choice(DEVICES),
        "app_version": f"{rng.randint(1,5)}.{rng.randint(0,12)}.{rng.randint(0,20)}",
    }

def rand_network(rng: random.Random) -> Dict[str, Any]:
    return {
        "network": rng.choice(NETWORKS),
        "ip": f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
        "asn": rng.randint(10000, 99999),
    }

def rand_policy_hints(rng: random.Random) -> Dict[str, Any]:
    algos = rng.sample(ALGOS, rng.randint(1, min(3, len(ALGOS))))
    return {
        "preferred_algorithms": [a["name"] for a in algos],
        "require_qkd": any(a["name"] == "QKD" for a in algos) and rng.random() < 0.5,
        "delivery_method_hint": rng.choice(DELIVERY_HINTS),
    }

def rand_pii_flags(rng: random.Random, email: str, phone: Tuple[int,int,int], card: str) -> Dict[str, Any]:
    p1, p2, p3 = phone
    return {
        "contains_email": rng.random() < 0.6,
        "contains_phone": rng.random() < 0.5,
        "contains_card": rng.random() < 0.4,
        "contains_secret": rng.random() < 0.3,
        "summary": {
            "emails": [email] if rng.random() < 0.6 else [],
            "phones": [f"+1-{p1}-{p2}-{p3}"] if rng.random() < 0.5 else [],
            "cards": [card] if rng.random() < 0.4 else [],
        }
    }

def rand_request_id() -> str:
    return str(uuid.uuid4())

def rand_traceparent(rng: random.Random) -> str:
    trace_id = uuid.uuid4().hex
    parent_id = rng.getrandbits(64).to_bytes(8, "big").hex()
    return f"00-{trace_id}-{parent_id}-01"

def iso_now_ms() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"

def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def maybe_longer_text(rng: random.Random, base: str) -> str:
    repeats = rng.choices([1, 2, 3, 5, 8], weights=[0.6, 0.2, 0.1, 0.07, 0.03], k=1)[0]
    return "\n\n".join([base] * repeats)

# ---------- Geradores de dados ----------

def rand_message(rng: random.Random) -> str:
    year = rng.choice([2024, 2025])
    q = rng.randint(1, 4)
    sprint = rng.randint(10, 99)
    phase = rng.choice(["I", "II", "III", "IV"])
    cid = rng.randint(100, 999)
    eid = rng.randint(1000, 9999)
    email = rand_email(rng)
    p1, p2, p3 = rand_phone(rng)
    pwd = rng.choice(PASSWORDS)
    secret = rng.choice(SECRETS)
    card = rng.choice(CARDS)
    pix = rng.choice(PIX_KEYS)
    count = rng.randint(1, 1000)

    lang = rng.choice(["en", "pt"])
    public_pool = PUBLIC_TEMPLATES_EN if lang == "en" else PUBLIC_TEMPLATES_PT

    buckets = [
        rng.choice(public_pool).format(q=q, year=year),
        rng.choice(INTERNAL_TEMPLATES).format(sprint=sprint, phase=phase),
        rng.choice(CONFIDENTIAL_TEMPLATES).format(cid=cid, eid=eid, email=email, p1=p1, p2=p2, p3=p3),
        rng.choice(RESTRICTED_TEMPLATES).format(cid=cid, pwd=pwd, secret=secret, card=card, pix=pix),
        rng.choice(NOISE_TEMPLATES).format(email=email, count=count),
    ]
    weights = [0.22, 0.22, 0.23, 0.20, 0.13]
    text = rng.choices(buckets, weights=weights, k=1)[0]
    return maybe_longer_text(rng, text)

def rand_metadata(rng: random.Random, message: str) -> Dict[str, Any]:
    lang, locale = rand_locale(rng)
    user = rand_user(rng)
    device = rand_device_info(rng)
    net = rand_network(rng)

    user_label = rng.choice(USER_LABELS) if rng.random() < 0.6 else None

    sample_email = user["email"]
    sample_phone = rand_phone(rng)
    sample_card = rng.choice(CARDS)
    pii = rand_pii_flags(rng, sample_email, sample_phone, sample_card)

    policy = rand_policy_hints(rng)

    content_type = rng.choice(["text/plain", "text/markdown", "application/json"])
    encoding = rng.choice(["utf-8", "utf-16", "latin-1"])
    content_hash = compute_hash(message)

    meta: Dict[str, Any] = {
        "doc_type": rng.choice(DOC_TYPES),
        "app": rng.choice(APPS),
        "severity": rng.choice(SEVERITIES),
        "classification": user_label,
        "language": lang,
        "locale": locale,
        "region": rng.choice(REGIONS),
        "timestamp": iso_now_ms(),
        "expiresAt": int(time.time()) + rng.randint(60, 3600),  # 1min a 1h
        "user": user,
        "device": device,
        "network": net,
        "pii": pii,
        "policy_hints": policy,
        "compliance_tags": rng.sample(
            ["GDPR", "PCI", "SOX", "HIPAA", "LGPD", "ISO27001"], rng.randint(0, 3)
        ),
        "content": {
            "type": content_type,
            "encoding": encoding,
            "length": len(message),
            "hash_sha256": content_hash,
        },
        "labels": rng.sample(
            ["urgent", "todo", "follow-up", "investigate", "auto", "manual"], rng.randint(0, 3)
        ),
        "related_ids": [f"RID-{rng.randint(10000,99999)}" for _ in range(rng.randint(0, 3))],
        "score": round(rng.uniform(0, 1), 3),
    }

    if rng.random() < 0.35:
        meta["ticket_id"] = f"TKT-{rng.randint(10000,99999)}"
    if rng.random() < 0.25:
        meta["campaign"] = f"CMP-{rng.randint(100,999)}"
    if rng.random() < 0.25:
        meta["session_id"] = f"sess-{rng.randint(100000,999999)}"

    return meta

def rand_ids(rng: random.Random) -> Tuple[str, str]:
    if rng.random() < 0.6:
        source = f"device-{rng.randint(100,999)}"
    else:
        source = f"svc-{rng.randint(10,99)}"
    destination = rng.choice(DESTS)
    return source, destination

# ---------- Geração de casos inválidos (opcional) ----------

def maybe_corrupt_payload(rng: random.Random, payload: Dict[str, Any], invalid_rate: float) -> Dict[str, Any]:
    if rng.random() >= invalid_rate:
        return payload

    choice = rng.choice(["drop_field", "wrong_type", "empty_message", "huge_message", "bad_metadata"])
    corrupted = json.loads(json.dumps(payload))

    if choice == "drop_field":
        to_drop = rng.choice(["message", "sourceId", "destinationId", "metadata"])
        corrupted.pop(to_drop, None)

    elif choice == "wrong_type":
        pick = rng.choice(["message", "sourceId", "destinationId", "metadata"])
        corrupted[pick] = 12345

    elif choice == "empty_message":
        corrupted["message"] = ""

    elif choice == "huge_message":
        corrupted["message"] = corrupted.get("message", "x") * rng.randint(5000, 20000)

    elif choice == "bad_metadata":
        if "metadata" in corrupted:
            corrupted["metadata"]["content"]["hash_sha256"] = "not_a_hash"
            corrupted["metadata"]["score"] = "NaN"
            corrupted["metadata"]["pii"] = None

    return corrupted

# ---------- Worker async ----------

async def worker(session: aiohttp.ClientSession, url: str, idx: int, rng_seed: int, print_resp: bool, timeout: int, invalid_rate: float):
    rng = random.Random(rng_seed + idx)

    request_id = rand_request_id()
    traceparent = rand_traceparent(rng)

    message = rand_message(rng)
    sourceId, destinationId = rand_ids(rng)
    metadata = rand_metadata(rng, message)

    headers = {
        "Content-Type": "application/json",
        "X-Source-Id": sourceId,
        "X-Destination-Id": destinationId,
        "X-Request-Id": request_id,
        "traceparent": traceparent,
        "X-Tenant-Id": metadata["user"].get("tenant_id") or "public",
    }

    payload = {
        "message": message,
        "sourceId": sourceId,
        "destinationId": destinationId,
        "metadata": {
            **metadata,
            "requestId": request_id,
        },
    }

    if rng.random() < 0.15:
        payload["metadata"]["debug"] = {
            "note": "extra fields for compatibility tests",
            "flags": [rng.choice(["A", "B", "C"]) for _ in range(rng.randint(0, 3))],
        }

    payload = maybe_corrupt_payload(rng, payload, invalid_rate)

    try:
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
            status = resp.status
            text = await resp.text()
            if print_resp:
                print(f"[{idx}] {status} {text[:300]}")
            return status, text
    except Exception as e:
        if print_resp:
            print(f"[{idx}] ERROR: {e}")
        return None, str(e)

# ---------- Agendador com controle de RPS ----------

async def run_load(url: str, total: int, concurrency: int, rps: float, timeout: int, seed: int, print_resp: bool, invalid_rate: float):
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        interval = 1.0 / rps if rps > 0 else 0
        tasks = []
        start = time.time()

        async def schedule_task(i: int):
            async with sem:
                if interval > 0:
                    await asyncio.sleep(i * interval / max(1, concurrency))  # espalha no tempo
                return await worker(session, url, i, seed, print_resp, timeout, invalid_rate)

        for i in range(total):
            tasks.append(asyncio.create_task(schedule_task(i)))

        results = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            completed += 1
            if completed % max(1, total // 10) == 0:
                ok = sum(1 for s, _ in results if s and 200 <= s < 300)
                err = sum(1 for s, _ in results if (s is None) or s >= 400)
                print(f"Progress: {completed}/{total} | OK={ok} ERR={err}")

        elapsed = time.time() - start
        ok = sum(1 for s, _ in results if s and 200 <= s < 300)
        err = sum(1 for s, _ in results if (s is None) or s >= 400)
        print(f"Done in {elapsed:.2f}s | total={total} OK={ok} ERR={err} (~{total/max(0.001,elapsed):.1f} req/s)")

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Load generator for /intercept with diverse payloads")
    parser.add_argument("--url", default="http://localhost:8080/intercept", help="Target URL")
    parser.add_argument("--total", type=int, default=200, help="Total de requisições")
    parser.add_argument("--concurrency", type=int, default=20, help="Número de conexões simultâneas")
    parser.add_argument("--rps", type=float, default=50, help="Requests por segundo (aprox.)")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout por requisição (s)")
    parser.add_argument("--seed", type=int, default=42, help="Seed de aleatoriedade")
    parser.add_argument("--invalid-rate", type=float, default=0.05, help="Proporção de payloads inválidos [0..1]")
    parser.add_argument("--print-resp", action="store_true", help="Imprime respostas (truncadas)")
    args = parser.parse_args()

    asyncio.run(run_load(
        url=args.url,
        total=args.total,
        concurrency=args.concurrency,
        rps=args.rps,
        timeout=args.timeout,
        seed=args.seed,
        print_resp=args.print_resp,
        invalid_rate=max(0.0, min(1.0, args.invalid_rate)),
    ))

if __name__ == "__main__":
    main()