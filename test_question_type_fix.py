#!/usr/bin/env python3
"""Test script to verify the question type fix."""

import json
import requests

def test_question_type_enforcement():
    """Test that API strictly enforces requested question type and difficulty."""
    
    print("🧪 Testing Question Type Enforcement Fix...")
    print("-" * 50)
    
    # Test data
    test_data = {
        'learning_objective': 'Understand basic neural network concepts',
        'content_summary': 'A neural network consists of layers of neurons that process information.',
        'question_type': 'multiple-choice',
        'difficulty_level': 'easy'
    }
    
    try:
        # Make API call
        response = requests.post('http://localhost:8000/questions/generate', json=test_data)
        response.raise_for_status()
        
        result = response.json()
        question = result['question']
        
        print(f"📋 RESULTS:")
        print(f"  Requested Type: {test_data['question_type']}")
        print(f"  Generated Type: {question['question_type']}")
        print(f"  ✅ Type Match: {question['question_type'] == test_data['question_type']}")
        print()
        print(f"  Requested Difficulty: {test_data['difficulty_level']}")
        print(f"  Generated Difficulty: {question['difficulty_level']}")
        print(f"  ✅ Difficulty Match: {question['difficulty_level'] == test_data['difficulty_level']}")
        print()
        
        # Check if fix worked
        type_fixed = question['question_type'] == test_data['question_type']
        difficulty_fixed = question['difficulty_level'] == test_data['difficulty_level']
        
        if type_fixed and difficulty_fixed:
            print("🎉 SUCCESS: LLM now follows user specifications exactly!")
        else:
            print("❌ STILL BROKEN: LLM is still overriding user specifications")
            
        print(f"\n📄 Generated Question: {question['question_text']}")
        print(f"💰 Cost: ${result['generation_cost']:.4f}")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: API server not running. Start with: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_question_type_enforcement() 