import json
import random
import time
import asyncio
import httpx
from datetime import datetime

API_URL = "http://192.168.18.18:8090"

def mutate_payload(base_payload):
    """Injects intentional noise or adversarial patterns."""
    mutated = base_payload.copy()
    if "data" in mutated:
        # 10% chance to corrupt a field
        if random.random() < 0.1:
            key = random.choice(list(mutated["data"].keys()))
            mutated["data"][key] = f"ADVERSARIAL_NOISE_{random.randint(1000, 9999)}"
        
        # 20% chance to add security-sensitive keywords to test BERT
        if random.random() < 0.2:
            mutated["data"]["anomaly_signature"] = "BASE64_SHELL_EXEC_DROPPER"
    
    return mutated

async def run_fuzzer(iterations=20):
    print(f"--- Starting Q-OPSEC Adversarial Fuzzer ({iterations} iterations) ---")
    
    # Load scenarios
    with open("/home/umbrel/projetos/Q-OPSEC/tests/pipeline_scenarios.json", "r") as f:
        scenarios = json.load(f)["scenarios"]

    results_log = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(iterations):
            scenario = random.choice(scenarios)
            test_id = f"fuzz-{datetime.now().strftime('%H%M%S')}-{i}"
            
            payload = mutate_payload(scenario["payload"])
            payload["request_id"] = test_id
            
            print(f"[{i+1}/{iterations}] Testing Scenario: {scenario['name']} | ID: {test_id}")
            
            try:
                start_t = time.time()
                response = await client.post(f"{API_URL}/run-pipeline", json=payload)
                end_t = time.time()
                
                if response.status_code == 200:
                    resp_data = response.json()
                    risk = resp_data.get("final_data", {}).get("risk_score", 0)
                    label = resp_data.get("final_data", {}).get("risk_label", "Unknown")
                    
                    results_log.append({
                        "id": test_id,
                        "scenario": scenario["id"],
                        "risk_score": risk,
                        "label": label,
                        "latency": end_t - start_t
                    })
                else:
                    print(f"  !! Pipeline Error: {response.status_code}")
            except Exception as e:
                print(f"  !! Connection Lost: {e}")
            
            await asyncio.sleep(0.5)

    # Save results for plotting
    output_path = "/home/umbrel/projetos/Q-OPSEC/logs/fuzzer_results.json"
    with open(output_path, "w") as f:
        json.dump(results_log, f, indent=2)
    
    print(f"\n--- Fuzzing Complete. Results saved to {output_path} ---")

if __name__ == "__main__":
    asyncio.run(run_fuzzer())
