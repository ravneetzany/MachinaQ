#!/usr/bin/env python
"""Quick test of enhanced model."""
import json
import urllib.request

url = "http://127.0.0.1:8001/analyze"
payload = {"step_path": "data/test_hole.step"}

try:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    with urllib.request.urlopen(req, timeout=10) as response:
        result = json.loads(response.read().decode('utf-8'))
        
    preds = result.get('predictions', [])
    print(f"\n✓ Enhanced Model Test Results")
    print(f"  Predictions: {len(preds)} features detected")
    
    if preds:
        for i, pred in enumerate(preds[:3], 1):
            print(f"  Feature {i}: {pred['predicted_type']:8s} | "
                  f"Confidence: {pred['confidence']:6.2%} | "
                  f"Face ID: {pred['face_id']}")
    
    print(f"\n  Model Status: {'✓ Available' if result.get('model_available') else '✗ Not available'}")
    print(f"  Summary: {result.get('summary', {})}")
    
except Exception as e:
    print(f"✗ Error: {e}")
