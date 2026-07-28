import httpx
import asyncio
import json

API_URL = "http://192.168.18.18:8090"

async def test_baseline_pipeline():
    print("Testing Baseline Pipeline...")
    payload = {
        "request_id": "test-baseline-001",
        "source": "automated-test",
        "data": {
            "message": "Baseline Test",
            "researcher": "Hermes Agent"
        }
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(f"{API_URL}/run-pipeline", json=payload)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Pipeline Result: {result.get('status', 'no status')}")
                # Verifica se os steps vitais passaram
                trace = result.get("pipeline_trace", [])
                for step in trace:
                    print(f" - {step['service']}: {step['status']}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Failed to connect: {e}")

if __name__ == "__main__":
    asyncio.run(test_baseline_pipeline())
