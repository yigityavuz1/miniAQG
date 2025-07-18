#!/usr/bin/env python3
"""
Simple API test script to debug the AQG API endpoints.
"""

import requests
import json
import sys

API_BASE = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_videos():
    """Test the videos endpoint."""
    print("\n🔍 Testing videos endpoint...")
    try:
        response = requests.get(f"{API_BASE}/videos")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Found {len(data['videos'])} videos")
        for video in data['videos'][:3]:  # Show first 3
            print(f"  - {video['video_id']}: {video['title']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_question_generation():
    """Test the question generation endpoint."""
    print("\n🔍 Testing question generation endpoint...")
    try:
        payload = {
            "learning_objective": "Understand the basic concept of neural networks",
            "content_summary": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections.",
            "question_type": "multiple-choice",
            "difficulty_level": "medium"
        }
        
        response = requests.post(
            f"{API_BASE}/questions/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Generated question: {data['question']['question_text'][:100]}...")
            print(f"Cost: ${data['generation_cost']:.4f}")
            print(f"Tokens: {data['tokens_used']}")
        else:
            print(f"Error response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing AQG API Endpoints")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Videos List", test_videos),
        ("Question Generation", test_question_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅' if result else '❌'} {test_name}: {'PASS' if result else 'FAIL'}")
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the API logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 