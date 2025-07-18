"""
FastAPI Application for AQG System

This module implements the REST API endpoints for the Automated Question Generation system,
providing access to individual agent functionalities and full workflow execution.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .models import (
    QuestionGenerateRequest, QuestionGenerateResponse,
    QuestionEvaluateRequest, QuestionEvaluateResponse,
    QuestionFixRequest, QuestionFixResponse,
    WorkflowRunRequest, WorkflowRunResponse,
    WorkflowStatusResponse, VideoResultsResponse,
    APIError, HealthCheckResponse
)
from ..workflows.aqg_workflow import AQGWorkflow
from ..workflows.state_models import Question, JudgeEvaluation, FixerAttempt
from ..core.llm_client import LLMClient
from ..core.prompt_loader import PromptLoader
from ..core.logger import AQGLogger


# ============================================================================
# FastAPI App Configuration
# ============================================================================

app = FastAPI(
    title="AQG System API",
    description="REST API for the Automated Question Generation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State and Dependencies
# ============================================================================

# In-memory storage for workflow sessions (in production, use Redis or database)
active_workflows: Dict[str, Dict[str, Any]] = {}
workflow_results: Dict[str, Any] = {}

# Initialize core components
llm_client = LLMClient()
prompt_loader = PromptLoader()
logger = AQGLogger()


# ============================================================================
# Helper Functions
# ============================================================================

async def initialize_components():
    """Initialize LLM client and other async components."""
    try:
        await llm_client.initialize()
        logger.info("API components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize API components: {e}")
        return False


def get_video_transcript(video_id: str) -> str:
    """Load transcript for a given video ID."""
    transcript_path = Path(f"transcripts/json/{video_id}.json")
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail=f"Transcript not found for video {video_id}")
    
    # Load JSON transcript and extract text
    import json
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Extract text from segments
    transcript_text = ""
    for segment in transcript_data.get("segments", []):
        transcript_text += segment.get("text", "") + " "
    
    return transcript_text.strip()


def get_video_title(video_id: str) -> str:
    """Get video title for a given video ID."""
    # This could be enhanced to read from a metadata file
    video_titles = {
        "aircAruvnKk": "But what is a neural network?",
        "IIB1Pdlhr4w": "Gradient descent, how neural networks learn",
        "Ilg3gGewQ5U": "What is backpropagation really doing?",
        # Add more video mappings as needed
    }
    return video_titles.get(video_id, f"Video {video_id}")


# ============================================================================
# Core Agent Endpoints
# ============================================================================

@app.post("/questions/generate", response_model=QuestionGenerateResponse)
async def generate_question(request: QuestionGenerateRequest):
    """Generate a single question based on learning objective and content summary."""
    try:
        start_time = time.time()
        
        # Load question generator prompt
        prompt_template = prompt_loader.load_prompt("question_generator", "v1")
        
        # Format prompt - ENFORCE user specifications
        prompt = prompt_template.replace("{learning_objective}", request.learning_objective)
        prompt = prompt.replace("{content_summary}", request.content_summary)
        prompt = prompt.replace("{content_segment}", "N/A")  # Not available in single request
        prompt = prompt.replace("{requested_question_type}", request.question_type)
        prompt = prompt.replace("{requested_difficulty}", request.difficulty_level)
        
        # Call LLM
        response = await llm_client.call_model(
            prompt=prompt,
            model="gpt-4o"
        )
        
        # Parse response
        try:
            question_data = json.loads(response.content)
            
            # Handle different response formats
            if "questions" in question_data:
                q_data = question_data["questions"][0]
            else:
                q_data = question_data
            
            # Handle answer options that might be in dict format
            raw_options = q_data.get("options", q_data.get("answer_options", []))
            answer_options = []
            if raw_options and isinstance(raw_options[0], dict):
                # Extract text from dict format: {"option_id": "A", "option_text": "..."}
                answer_options = [opt.get("option_text", str(opt)) for opt in raw_options]
            else:
                # Already in string format
                answer_options = raw_options
            
            # Create Question object - STRICTLY enforce user specifications
            question = Question(
                id=f"api_q_{int(time.time())}",
                question_type=request.question_type,  # ALWAYS use requested type!
                question_text=q_data.get("question_text", ""),
                correct_answer=q_data.get("correct_answer", ""),
                answer_options=answer_options,
                learning_objective_assessed=request.learning_objective,
                learning_objective_alignment=request.learning_objective,
                difficulty_level=request.difficulty_level,  # ALWAYS use requested difficulty!
                rationale_for_judge=q_data.get("rationale_for_judge", "")
            )
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            question = Question(
                id=f"api_q_{int(time.time())}",
                question_type=request.question_type,
                question_text="Failed to parse question from LLM response",
                correct_answer="N/A",
                answer_options=[],
                learning_objective_assessed=request.learning_objective,
                learning_objective_alignment=request.learning_objective,
                difficulty_level=request.difficulty_level,
                rationale_for_judge="Generated via API"
            )
        
        generation_time = time.time() - start_time
        
        return QuestionGenerateResponse(
            question=question,
            generation_cost=response.usage.cost_usd,
            generation_time=generation_time,
            tokens_used=response.usage.total_tokens
        )
        
    except Exception as e:
        logger.error(f"Error generating question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/questions/evaluate", response_model=QuestionEvaluateResponse)
async def evaluate_question(request: QuestionEvaluateRequest):
    """Evaluate a single question using the judge agent."""
    try:
        start_time = time.time()
        
        # Load judge prompt
        prompt_template = prompt_loader.load_prompt("judge", "v2")  # Using balanced judge
        
        # Format prompt - Create properly structured question object
        question = request.question
        question_data = {
            "id": question.id,
            "question_type": question.question_type,
            "question_text": question.question_text,
            "correct_answer": question.correct_answer,
            "answer_options": question.answer_options,
            "learning_objective_assessed": request.learning_objective or question.learning_objective_assessed,
            "difficulty_level": question.difficulty_level,
            "rationale_for_judge": question.rationale_for_judge
        }
        
        # Format the prompt with the question data
        prompt = prompt_template.replace("{questions}", json.dumps([question_data], indent=2))
        
        # Call LLM
        response = await llm_client.call_model(
            prompt=prompt,
            model="gpt-4o"
        )
        
        # Parse response
        try:
            eval_data = json.loads(response.content)
            
            # Handle different response formats
            if "evaluations" in eval_data:
                eval_data = eval_data["evaluations"][0]
            
            decision = eval_data.get("decision", "Rejected")
            evaluation = JudgeEvaluation(
                question_id=question.id,
                decision=decision,
                pass_status=(decision.lower() == "approved"),
                evaluation_details=eval_data.get("evaluation_details", []),
                overall_feedback_summary=eval_data.get("overall_feedback_summary", "")
            )
            
        except json.JSONDecodeError:
            # Fallback evaluation
            evaluation = JudgeEvaluation(
                question_id=question.id,
                decision="Error",
                pass_status=False,
                evaluation_details=[],
                overall_feedback_summary="Failed to parse judge response"
            )
        
        evaluation_time = time.time() - start_time
        
        return QuestionEvaluateResponse(
            evaluation=evaluation,
            evaluation_cost=response.usage.cost_usd,
            evaluation_time=evaluation_time,
            tokens_used=response.usage.total_tokens
        )
        
    except Exception as e:
        logger.error(f"Error evaluating question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/questions/fix", response_model=QuestionFixResponse)
async def fix_question(request: QuestionFixRequest):
    """Fix a rejected question using the fixer agent."""
    try:
        start_time = time.time()
        
        # Load fixer prompt
        prompt_template = prompt_loader.load_prompt("fixer", "v1")
        
        # Format prompt
        question = request.rejected_question
        evaluation = request.evaluation
        
        prompt = prompt_template.replace("{original_question}", question.question_text)
        prompt = prompt.replace("{question_type}", str(question.question_type))
        prompt = prompt.replace("{correct_answer}", str(question.correct_answer))
        prompt = prompt.replace("{answer_options}", str(question.answer_options))
        prompt = prompt.replace("{judge_feedback}", evaluation.overall_feedback_summary)
        prompt = prompt.replace("{suggested_improvements}", evaluation.overall_feedback_summary)
        prompt = prompt.replace("{criteria_scores}", str(evaluation.evaluation_details))
        
        # Call LLM
        response = await llm_client.call_model(
            prompt=prompt,
            model="gpt-4o"
        )
        
        # Parse response
        try:
            fix_data = json.loads(response.content)
            
            # Create fixed question
            fixed_question = Question(
                id=question.id,
                question_type=question.question_type,
                question_text=fix_data.get("fixed_question", question.question_text),
                correct_answer=fix_data.get("fixed_answer", question.correct_answer),
                answer_options=fix_data.get("fixed_options", question.answer_options),
                learning_objective_assessed=question.learning_objective_assessed,
                learning_objective_alignment=question.learning_objective_alignment,
                difficulty_level=question.difficulty_level,
                rationale_for_judge=fix_data.get("improved_rationale", question.rationale_for_judge),
                judge_evaluations=question.judge_evaluations.copy(),
                fixer_attempts=question.fixer_attempts.copy()
            )
            
            # Add this fix attempt
            fixer_attempt = FixerAttempt(
                attempt_number=request.attempt_number,
                original_question=question.question_text,
                fixed_question=fix_data.get("fixed_question", ""),
                changes_made=fix_data.get("changes_made", []),
                reasoning=fix_data.get("reasoning", ""),
                judge_feedback=evaluation
            )
            fixed_question.fixer_attempts.append(fixer_attempt)
            
            changes_made = fix_data.get("changes_made", [])
            reasoning = fix_data.get("reasoning", "")
            
        except json.JSONDecodeError:
            # Fallback fix
            fixed_question = question
            changes_made = ["Failed to parse fixer response"]
            reasoning = "Error occurred during fixing"
        
        fix_time = time.time() - start_time
        
        return QuestionFixResponse(
            fixed_question=fixed_question,
            changes_made=changes_made,
            fix_reasoning=reasoning,
            fix_cost=response.usage.cost_usd,
            fix_time=fix_time,
            tokens_used=response.usage.total_tokens
        )
        
    except Exception as e:
        logger.error(f"Error fixing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Workflow Endpoints
# ============================================================================

async def run_workflow_background(session_id: str, video_id: str, max_iterations: int = 5):
    """Background task to run the full AQG workflow."""
    try:
        # Update workflow status
        active_workflows[session_id]["status"] = "running"
        active_workflows[session_id]["current_stage"] = "initializing"
        
        # Load transcript and get title
        transcript = get_video_transcript(video_id)
        video_title = get_video_title(video_id)
        
        # Initialize workflow
        workflow = AQGWorkflow(llm_client, prompt_loader, logger)
        
        # Update status
        active_workflows[session_id]["current_stage"] = "content_splitting"
        
        # Run workflow
        result = await workflow.process_video(video_id, video_title, transcript)
        
        # Store results
        workflow_results[session_id] = result
        active_workflows[session_id]["status"] = "completed"
        active_workflows[session_id]["result"] = result
        
        logger.info(f"Workflow {session_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Workflow {session_id} failed: {e}")
        active_workflows[session_id]["status"] = "failed"
        active_workflows[session_id]["error"] = str(e)


@app.post("/workflow/run/{video_id}", response_model=WorkflowRunResponse)
async def run_workflow(video_id: str, request: Optional[WorkflowRunRequest] = None, background_tasks: BackgroundTasks = BackgroundTasks()):
    """Trigger the full AQG workflow for a video."""
    try:
        # Generate session ID
        session_id = f"workflow_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Initialize workflow tracking
        active_workflows[session_id] = {
            "video_id": video_id,
            "status": "started",
            "start_time": datetime.now(),
            "current_stage": "initializing",
            "max_iterations": request.max_iterations if request else 5
        }
        
        # Start background task
        background_tasks.add_task(run_workflow_background, session_id, video_id, 
                                request.max_iterations if request else 5)
        
        return WorkflowRunResponse(
            session_id=session_id,
            video_id=video_id,
            status="started",
            estimated_time="2-5 minutes"
        )
        
    except Exception as e:
        logger.error(f"Error starting workflow for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/status/{session_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(session_id: str):
    """Get the status of a running workflow."""
    if session_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow session not found")
    
    workflow_info = active_workflows[session_id]
    
    # Calculate progress percentage based on stage
    stage_progress = {
        "initializing": 0,
        "content_splitting": 10,
        "summarization": 30,
        "question_generation": 50,
        "judging": 70,
        "fixing": 85,
        "completed": 100,
        "failed": 0
    }
    
    current_stage = workflow_info.get("current_stage", "initializing")
    progress = stage_progress.get(current_stage, 0)
    
    # If we have results, extract metrics
    questions_generated = 0
    questions_approved = 0
    questions_rejected = 0
    total_cost = 0.0
    
    if "result" in workflow_info:
        result = workflow_info["result"]
        questions_generated = result.total_questions_generated
        questions_approved = result.approved_questions
        questions_rejected = result.rejected_questions
        total_cost = result.total_cost
    
    return WorkflowStatusResponse(
        session_id=session_id,
        status=workflow_info["status"],
        current_stage=current_stage,
        progress_percentage=progress,
        total_cost=total_cost,
        questions_generated=questions_generated,
        questions_approved=questions_approved,
        questions_rejected=questions_rejected,
        estimated_completion=None,  # Could be calculated based on progress
        error_message=workflow_info.get("error")
    )


@app.get("/results/{video_id}", response_model=VideoResultsResponse)
async def get_video_results(video_id: str):
    """Retrieve the final results of a completed workflow."""
    # Look for results file
    results_file = Path(f"outputs/{video_id}_results.json")
    
    if not results_file.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for video {video_id}")
    
    try:
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Convert to VideoProcessingResult (this might need adjustment based on the actual structure)
        return VideoResultsResponse(
            video_id=video_id,
            results=results_data,  # In a real implementation, you'd properly parse this
            results_file_path=str(results_file)
        )
        
    except Exception as e:
        logger.error(f"Error loading results for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading results: {e}")


# ============================================================================
# Health and Utility Endpoints
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check dependencies
        dependencies = {
            "llm_client": "healthy" if llm_client.openai_client else "error",
            "prompt_loader": "healthy",
            "logger": "healthy"
        }
        
        status = "healthy" if all(dep == "healthy" for dep in dependencies.values()) else "degraded"
        
        return HealthCheckResponse(
            status=status,
            version="1.0.0",
            dependencies=dependencies
        )
        
    except Exception as e:
        return HealthCheckResponse(
            status="error",
            version="1.0.0",
            dependencies={"error": str(e)}
        )


@app.get("/videos")
async def list_videos():
    """List available videos for processing."""
    transcript_dir = Path("transcripts/json")
    if not transcript_dir.exists():
        return {"videos": []}
    
    video_files = list(transcript_dir.glob("*.json"))
    videos = []
    
    for video_file in video_files:
        video_id = video_file.stem
        videos.append({
            "video_id": video_id,
            "title": get_video_title(video_id),
            "transcript_path": str(video_file)
        })
    
    return {"videos": videos}


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "details": {"status_code": exc.status_code},
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "details": {"exception_type": type(exc).__name__},
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting AQG API server...")
    
    success = await initialize_components()
    if not success:
        logger.error("Failed to initialize API components")
        # In production, you might want to exit here
    else:
        logger.info("AQG API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down AQG API server...")
    # Clean up any resources here


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 