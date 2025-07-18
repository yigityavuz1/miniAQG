#!/usr/bin/env python3
"""Test script to verify the judge endpoint fix."""

import json
import requests

def test_judge_evaluation_fix():
    """Test that judge evaluates the actual question provided, not hardcoded examples."""
    
    print("🧪 Testing Judge Evaluation Fix...")
    print("-" * 60)
    
    # Test data - activation function question
    test_data = {
        "question": {
            "id": "test_q_1",
            "question_type": "multiple-choice",
            "question_text": "What is the primary function of an activation function in a neural network?",
            "correct_answer": "To introduce non-linearity into the network",
            "answer_options": [
                "To store the network weights",
                "To introduce non-linearity into the network", 
                "To calculate the loss function",
                "To optimize the learning rate"
            ],
            "learning_objective_assessed": "Understand activation functions",
            "learning_objective_alignment": "Understand activation functions",
            "difficulty_level": "medium",
            "rationale_for_judge": "This question tests understanding of activation functions."
        },
        "learning_objective": "Understand activation functions in neural networks"
    }
    
    try:
        # Make API call
        response = requests.post('http://localhost:8000/questions/evaluate', json=test_data)
        response.raise_for_status()
        
        result = response.json()
        evaluation = result['evaluation']
        
        print("📋 EVALUATION RESULTS:")
        print(f"  Question ID: {evaluation['question_id']}")
        print(f"  Decision: {evaluation['decision']}")
        print(f"  Pass Status: {evaluation['pass_status']}")
        print()
        
        print("📝 EVALUATION DETAILS:")
        for detail in evaluation['evaluation_details']:
            print(f"  • {detail['criterion_name']}: {detail['status']}")
            print(f"    Feedback: {detail['feedback']}")
            print()
        
        print("📄 OVERALL FEEDBACK:")
        print(f"  {evaluation['overall_feedback_summary']}")
        print()
        
        # Check if fix worked - should reference activation functions, not arithmetic
        feedback_text = evaluation['overall_feedback_summary'].lower()
        details_text = " ".join([d['feedback'] for d in evaluation['evaluation_details']]).lower()
        all_feedback = feedback_text + " " + details_text
        
        # Check for correct content
        activation_mentioned = any(term in all_feedback for term in ['activation', 'neural network', 'non-linearity'])
        arithmetic_mentioned = any(term in all_feedback for term in ['arithmetic', '2 + 2', 'addition'])
        
        print("🔍 CONTENT ANALYSIS:")
        print(f"  ✅ Mentions activation functions/neural networks: {activation_mentioned}")
        print(f"  ❌ Mentions arithmetic/2+2: {arithmetic_mentioned}")
        print()
        
        if activation_mentioned and not arithmetic_mentioned:
            print("🎉 SUCCESS: Judge now evaluates the ACTUAL question provided!")
        elif arithmetic_mentioned:
            print("❌ STILL BROKEN: Judge is still using hardcoded arithmetic examples")
        else:
            print("⚠️  UNCLEAR: Judge feedback doesn't clearly reference the question content")
            
        print(f"\n💰 Cost: ${result['evaluation_cost']:.4f}")
        print(f"⏱️  Time: {result['evaluation_time']:.2f}s")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: API server not running. Start with: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_judge_evaluation_fix() 