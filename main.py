#!/usr/bin/env python3
"""
Mini AQG (Automated Question Generation) System - Main Entry Point
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.core import get_logger, prompt_loader
from src.core.llm_client import LLMClient
from src.workflows.state_models import AQGState, ProcessingStage
from transcribe import get_available_videos


def show_system_status():
    """Show current system status and capabilities."""
    print("🚀 Mini AQG System Status")
    print("=" * 50)
    
    # Check prompt system
    prompt_summary = prompt_loader.get_prompt_summary()
    print(f"✅ Prompt System: {prompt_summary['total_agents']} agents configured")
    
    # Check for missing prompt files
    if prompt_summary['missing_files']:
        print(f"⚠️  Missing prompt files: {len(prompt_summary['missing_files'])}")
        for missing in prompt_summary['missing_files'][:3]:
            print(f"   - {missing}")
        if len(prompt_summary['missing_files']) > 3:
            print(f"   ... and {len(prompt_summary['missing_files']) - 3} more")
    
    # Check transcript availability
    try:
        videos = get_available_videos()
        print(f"✅ Transcripts: {len(videos)} videos available")
        
        # Show available videos
        if videos:
            print("\n📹 Available Videos:")
            for video in videos[:5]:  # Show first 5
                print(f"   - {video['id']}: {video['title'][:60]}...")
            if len(videos) > 5:
                print(f"   ... and {len(videos) - 5} more")
    except Exception as e:
        print(f"⚠️  Transcripts: Error loading videos - {e}")
    
    # Check LLM configuration
    try:
        llm_client = LLMClient()
        cost_summary = llm_client.get_cost_summary()
        print(f"✅ LLM Client: Initialized")
        
        # Check API keys
        if llm_client.openai_client:
            print("   - OpenAI client: Ready")
        else:
            print("   - OpenAI client: ⚠️ API key missing")
            
        if llm_client.google_client:
            print("   - Google client: Ready")
        else:
            print("   - Google client: ⚠️ API key missing")
            
    except Exception as e:
        print(f"⚠️  LLM Client: Error - {e}")
    
    # Check logging
    logger = get_logger()
    print(f"✅ Logging: Session initialized")
    
    print("\n💡 Next Steps:")
    print("   1. Set up API keys in .env file")
    print("   2. Run demo: python main.py --demo")
    print("   3. Process a video: python main.py --video <video_id>")


def run_demo():
    """Run a demo of the AQG workflow."""
    print("🎯 Mini AQG System Demo")
    print("=" * 50)
    
    try:
        videos = get_available_videos()
        if not videos:
            print("No videos available for demo")
            return
        
        video = videos[0]
        print(f"Using video: {video['title']}")
        
        # Create demo state to show the workflow structure
        print(f"\n📋 Demo workflow stages:")
        print("   1. Content Splitter - Analyze transcript structure")
        print("   2. Summarizer - Generate learning objectives")
        print("   3. Question Generator - Create assessment questions")
        print("   4. Judge - Evaluate question quality")
        print("   5. Fixer - Revise rejected questions")
        
        # Show stage details
        stages = [
            ProcessingStage.CONTENT_SPLITTING,
            ProcessingStage.SUMMARIZATION,
            ProcessingStage.QUESTION_GENERATION,
            ProcessingStage.JUDGING,
            ProcessingStage.FIXING,
            ProcessingStage.COMPLETED
        ]
        
        print(f"\n📊 Workflow Structure:")
        for i, stage in enumerate(stages, 1):
            print(f"   {i}. {stage.value}")
        
        # Check prompt availability for each agent
        print(f"\n🔧 Agent Configuration:")
        agents = prompt_loader.list_agents()
        for agent in agents:
            config = prompt_loader.get_agent_config(agent)
            model = config.get('model', 'gpt-4o')
            provider = config.get('model_provider', 'openai')
            print(f"   - {agent}: {model} ({provider})")
        
        print(f"\n✅ Demo completed!")
        print(f"💰 Target cost per question: $0.05")
        print(f"🔄 Max judge iterations: 5")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def slugify(text):
    """Convert text to a safe filename slug (same as in transcribe.py)."""
    import re
    import unicodedata
    
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[\s]+', '_', text)
    return text


def find_transcript_file(video_id: str, videos: List[Dict]) -> Optional[str]:
    """Find the transcript filename for a given video ID."""
    # First try the video ID directly
    transcript_path = Path(f"transcripts/json/{video_id}.json")
    if transcript_path.exists():
        return str(transcript_path)
    
    # Try to find by slugified title
    for video in videos:
        if video['id'] == video_id:
            safe_title = slugify(video['title'])
            transcript_path = Path(f"transcripts/json/{safe_title}.json")
            if transcript_path.exists():
                return str(transcript_path)
    
    return None


def process_video(video_id: str):
    """Process a specific video through the AQG workflow."""
    print(f"🎯 Processing Video: {video_id}")
    print("=" * 50)
    
    try:
        # Get video information
        videos = get_available_videos()
        video = next((v for v in videos if v['id'] == video_id), None)
        
        if not video:
            print(f"❌ Video not found: {video_id}")
            return
        
        print(f"📹 Video: {video['title']}")
        
        # Find transcript file
        transcript_path = find_transcript_file(video_id, videos)
        if not transcript_path:
            print(f"❌ Transcript not found for video: {video_id}")
            print(f"   Checked: transcripts/json/{video_id}.json")
            print(f"   Checked: transcripts/json/{slugify(video['title'])}.json")
            return
        
        print(f"📄 Found transcript: {transcript_path}")
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Extract full text from captions
        captions = transcript_data.get('captions', [])
        if not captions:
            print("❌ Empty transcript")
            return
        
        transcript_text = ' '.join([caption.get('text', '') for caption in captions])
        
        if not transcript_text.strip():
            print("❌ Empty transcript text")
            return
        
        print(f"📄 Transcript loaded: {len(transcript_text)} characters, {len(captions)} segments")
        
        # Initialize workflow components
        logger = get_logger()
        llm_client = LLMClient(logger=logger)
        
        from src.workflows.aqg_workflow import AQGWorkflow
        workflow = AQGWorkflow(llm_client, logger)
        
        print("🚀 Starting AQG workflow...")
        
        # Process the video
        import asyncio
        
        async def run_workflow():
            result = await workflow.process_video(
                video_id=video_id,
                video_title=video['title'],
                transcript=transcript_text
            )
            return result
        
        # Run the workflow
        result = asyncio.run(run_workflow())
        
        # Display results
        print(f"\n✅ Workflow completed!")
        print(f"📊 Results:")
        print(f"   - Questions generated: {result.total_questions_generated}")
        print(f"   - Questions approved: {result.approved_questions}")
        print(f"   - Questions rejected: {result.rejected_questions}")
        print(f"   - Judge iterations: {result.judge_iterations}")
        print(f"   - Total cost: ${result.total_cost:.4f}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        
        # Cost analysis
        cost_per_question = result.total_cost / max(1, result.approved_questions)
        print(f"   - Cost per approved question: ${cost_per_question:.4f}")
        
        if cost_per_question <= 0.05:
            print("   ✅ Within target cost of $0.05 per question")
        else:
            print("   ⚠️ Exceeds target cost of $0.05 per question")
        
        # Save results
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        result_file = output_dir / f"{video_id}_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        
        print(f"📁 Results saved to: {result_file}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


def test_infrastructure():
    """Test basic infrastructure components."""
    print("🧪 Testing Mini AQG Infrastructure")
    print("=" * 50)
    
    try:
        # Test logger
        logger = get_logger()
        logger.info("Testing logger functionality")
        print("✅ Logger: Working")
        
        # Test prompt loader
        prompt_summary = prompt_loader.get_prompt_summary()
        print(f"✅ Prompt System: {prompt_summary['total_agents']} agents loaded")
        
        # Test LLM client
        llm_client = LLMClient(logger=logger)
        print("✅ LLM Client: Initialized successfully")
        
        # Test a simple prompt call
        import asyncio
        
        async def test_llm_call():
            try:
                # Load a simple prompt
                test_prompt = "Explain what a neural network is in one sentence."
                
                print("🔄 Testing LLM call...")
                response = await llm_client.call_model(
                    prompt=test_prompt,
                    model="gpt-4o",
                    system_instruction="You are a helpful AI assistant."
                )
                
                print(f"✅ LLM Call: Success")
                print(f"   📝 Response: {response.content[:100]}...")
                print(f"   💰 Cost: ${response.usage.cost_usd:.4f}")
                print(f"   🕐 Time: {response.response_time:.2f}s")
                
                return True
                
            except Exception as e:
                print(f"❌ LLM Call: Failed - {e}")
                return False
        
        # Run LLM test
        llm_success = asyncio.run(test_llm_call())
        
        if llm_success:
            print("\n🎉 All infrastructure tests passed!")
            print("\n💡 Next steps:")
            print("   1. Check prompt templates for proper JSON formatting")
            print("   2. Test with shorter video for workflow debugging")
            print("   3. Fix workflow state management issues")
        else:
            print("\n⚠️  LLM integration needs attention")
            
    except Exception as e:
        print(f"❌ Infrastructure test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mini AQG System")
    parser.add_argument("--demo", action="store_true", help="Run system demo")
    parser.add_argument("--video", type=str, help="Process specific video ID")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--list-videos", action="store_true", help="List available videos")
    parser.add_argument("--test", action="store_true", help="Test infrastructure components")
    
    args = parser.parse_args()
    
    if args.status:
        show_system_status()
    elif args.list_videos:
        try:
            videos = get_available_videos()
            print(f"📹 Available Videos ({len(videos)}):")
            for video in videos:
                print(f"   {video['id']}: {video['title']}")
        except Exception as e:
            print(f"Error: {e}")
    elif args.demo:
        run_demo()
    elif args.test:
        test_infrastructure()
    elif args.video:
        process_video(args.video)
    else:
        print("Mini AQG (Automated Question Generation) System")
        print("Usage:")
        print("  python main.py --status      # Show system status")
        print("  python main.py --demo        # Run demo workflow")
        print("  python main.py --list-videos # List available videos")
        print("  python main.py --video <id>  # Process specific video")
        print("  python main.py --test        # Test infrastructure")


if __name__ == "__main__":
    main() 