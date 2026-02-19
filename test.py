import torch
import httpx
from main import app
import asyncio

async def run_test():
    print("STARTING LOGIC TEST")
    async with httpx.AsyncClient(
    transport=httpx.ASGITransport(app=app, lifespan="on"),
    base_url="http://test") as client:

        
        # 1. Check Health 
        health_resp = await client.get("/health")
        print(f"HEALTH STATUS: {health_resp.status_code}")
        print(f"MODEL LOADED: {health_resp.json().get('model_loaded')}")

        # 2. Testing image gen
        print("\nTESTING IMAGE GEN")
        payload = {
            "prompt": "Image of a painting in style of gond painting of a group of people gathered around a large tree",
            "num_inference_steps": 2,
            "guidance_scale": 7,
        }
        
        response = await client.post("/generate", json=payload)

        if response.status_code == 200:
            data = response.json()
            print("SUCCESS")
            print(f"Used Seed: {data['seed']}")
        else:
            print(f"FAILED with Status: {response.status_code}")
            print(f"Error Detail: {response.text}")

if __name__ == "__main__":
    asyncio.run(run_test())