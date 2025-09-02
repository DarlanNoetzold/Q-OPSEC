#!/usr/bin/env python3
import asyncio
import aiohttp
import random
import argparse
import json
import time
from typing import Dict, Any

# ---------- Geradores de dados ----------

DOC_TYPES = ["finance", "hr", "legal", "marketing", "support", "engineering", "sales", "ops"]
APPS = ["mobile", "web", "admin", "partner", "api"]
DESTS = ["service-payments", "service-hr", "service-legal", "service-analytics", "service-support", "service-risk"]
SEVERITIES = ["low", "medium", "high"]

PUBLIC_TEMPLATES = [
    "Product update announcement for Q{q} {year}.",
    "Open API call for feedback on new feature beta.",
    "General notice to all users regarding terms update."
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
    "Deploy secret key {secret} to production vault immediately."
]

EMAIL_USERS = ["john.doe", "jane.smith", "admin", "finance.bot", "alerts"]
EMAIL_DOMAINS = ["example.com", "corp.local", "partner.net", "company.io"]

PASSWORDS = ["admin123", "P@ssw0rd", "Secret_2025!", "letmein", "Tr0ub4dor&3"]
SECRETS = ["sk_live_9f8a7s6d5f4", "AKIAIOSFODNN7EXAMPLE", "ghp_abc123xyz987", "l1v3-prod-KEY"]
CARDS = ["4111111111111111", "5555555555554444", "4000000000000002", "378282246310005"]  # testes

def rand_email(rng: random.Random) -> str:
    return f"{rng.choice(EMAIL_USERS)}@{rng.choice(EMAIL_DOMAINS)}"

def rand_phone(rng: random.Random):
    return rng.randint(201, 989), rng.randint(100, 999), rng.randint(1000, 9999)

def rand_message(rng: random.Random) -> str:
    year = rng.choice([2024, 2025])
    q = rng.randint(1, 4)
    sprint = rng.randint(10, 99)
    phase = rng.choice(["I", "II", "III"])
    cid = rng.randint(100, 999)
    eid = rng.randint(1000, 9999)
    email = rand_email(rng)
    p1, p2, p3 = rand_phone(rng)
    pwd = rng.choice(PASSWORDS)
    secret = rng.choice(SECRETS)
    card = rng.choice(CARDS)

    buckets = [
        rng.choice(PUBLIC_TEMPLATES).format(q=q, year=year),
        rng.choice(INTERNAL_TEMPLATES).format(sprint=sprint, phase=phase),
        rng.choice(CONFIDENTIAL_TEMPLATES).format(cid=cid, eid=eid, email=email, p1=p1, p2=p2, p3=p3),
        rng.choice(RESTRICTED_TEMPLATES).format(cid=cid, pwd=pwd, secret=secret, card=card),
    ]
    # Pondera para gerar mistura diversa
    weights = [0.25, 0.25, 0.3, 0.2]
    return rng.choices(buckets, weights=weights, k=1)[0]

def rand_metadata(rng: random.Random) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "doc_type": rng.choice(DOC_TYPES),
        "app": rng.choice(APPS),
        "severity": rng.choice(SEVERITIES),
    }
    # Opcionalmente inclua um rótulo de usuário
    if rng.random() < 0.3:
        meta["user_label"] = rng.choice(["public", "internal", "confidential", "restricted"])
    # Campos extras para enriquecer
    if rng.random() < 0.4:
        meta["region"] = rng.choice(["us", "eu", "latam", "apac"])
    if rng.random() < 0.25:
        meta["ticket_id"] = f"TKT-{rng.randint(10000,99999)}"
    return meta

def rand_ids(rng: random.Random):
    # source: um "device" ou "svc"
    if rng.random() < 0.6:
        source = f"device-{rng.randint(100,999)}"
    else:
        source = f"svc-{rng.randint(10,99)}"
    destination = rng.choice(DESTS)
    return source, destination

# ---------- Worker async ----------

async def worker(session: aiohttp.ClientSession, url: str, idx: int, rng_seed: int, print_resp: bool, timeout: int):
    rng = random.Random(rng_seed + idx)
    message = rand_message(rng)
    sourceId, destinationId = rand_ids(rng)
    metadata = rand_metadata(rng)

    headers = {
        "Content-Type": "application/json",
        "X-Source-Id": sourceId,
        "X-Destination-Id": destinationId,
    }
    payload = {
        "message": message,
        "sourceId": sourceId,
        "destinationId": destinationId,
        "metadata": metadata,
    }

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

async def run_load(url: str, total: int, concurrency: int, rps: float, timeout: int, seed: int, print_resp: bool):
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
                return await worker(session, url, i, seed, print_resp, timeout)

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
                err = sum(1 for s, _ in results if not s or s >= 400)
                print(f"Progress: {completed}/{total} | OK={ok} ERR={err}")

        elapsed = time.time() - start
        ok = sum(1 for s, _ in results if s and 200 <= s < 300)
        err = sum(1 for s, _ in results if not s or s >= 400)
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
    ))

if __name__ == "__main__":
    main()