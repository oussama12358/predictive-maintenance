#!/usr/bin/env python
"""Quick live test of the API endpoint"""
import httpx
import json

payload = {
    'machine_id': 'TEST-001',
    'product_type': 'M',
    'air_temp_c': 25.1,
    'process_temp_c': 36.4,
    'rotational_speed_rpm': 1551.0,
    'torque_nm': 42.8,
    'tool_wear_min': 108.0,
}

print("Testing /predict_failure endpoint...")
response = httpx.post('http://localhost:8000/predict_failure', json=payload, timeout=10)
print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))

print("\nTesting /health endpoint...")
response = httpx.get('http://localhost:8000/health', timeout=10)
print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))
