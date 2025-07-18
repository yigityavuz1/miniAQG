"""
API Models for AQG System

This module defines Pydantic models for API request and response bodies,
extending the core models from state_models.py for external API usage.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from ..workflows.state_models import Question, JudgeEvaluation, VideoProcessingResult


# ============================================================================
# Request Models
# ============================================================================

class QuestionGenerateRequest(BaseModel):
    """Request model for generating a single question."""
    learning_objective: str = Field(..., description="The learning objective to assess")
    content_summary: str = Field(..., description="Summary of the content to base the question on")
    question_type: Optional[str] = Field(default="multiple-choice", description="Type of question to generate")
    difficulty_level: Optional[str] = Field(default="medium", description="Difficulty level (easy, medium, hard)")


class QuestionEvaluateRequest(BaseModel):
    """Request model for evaluating a question."""
    question: Question = Field(..., description="The question to evaluate")
    learning_objective: Optional[str] = Field(None, description="Learning objective context for evaluation")


class QuestionFixRequest(BaseModel):
    """Request model for fixing a rejected question."""
    rejected_question: Question = Field(..., description="The question that was rejected")
    evaluation: JudgeEvaluation = Field(..., description="The judge evaluation with feedback")
    attempt_number: Optional[int] = Field(default=1, description="Which fix attempt this is")


class WorkflowRunRequest(BaseModel):
    """Request model for running a full workflow."""
    video_id: str = Field(..., description="ID of the video to process")
    max_iterations: Optional[int] = Field(default=5, description="Maximum judge/fixer iterations")
    custom_prompt_versions: Optional[Dict[str, str]] = Field(None, description="Custom prompt versions to use")


# ============================================================================
# Response Models
# ============================================================================

class QuestionGenerateResponse(BaseModel):
    """Response model for question generation."""
    question: Question = Field(..., description="The generated question")
    generation_cost: float = Field(..., description="Cost of generating this question")
    generation_time: float = Field(..., description="Time taken to generate the question")
    tokens_used: int = Field(..., description="Number of tokens used")


class QuestionEvaluateResponse(BaseModel):
    """Response model for question evaluation."""
    evaluation: JudgeEvaluation = Field(..., description="The judge evaluation")
    evaluation_cost: float = Field(..., description="Cost of evaluating the question")
    evaluation_time: float = Field(..., description="Time taken to evaluate")
    tokens_used: int = Field(..., description="Number of tokens used")


class QuestionFixResponse(BaseModel):
    """Response model for question fixing."""
    fixed_question: Question = Field(..., description="The fixed question")
    changes_made: List[str] = Field(..., description="List of changes made to the question")
    fix_reasoning: str = Field(..., description="Reasoning behind the fixes")
    fix_cost: float = Field(..., description="Cost of fixing the question")
    fix_time: float = Field(..., description="Time taken to fix")
    tokens_used: int = Field(..., description="Number of tokens used")


class WorkflowRunResponse(BaseModel):
    """Response model for workflow initiation."""
    session_id: str = Field(..., description="Unique session ID for tracking this workflow")
    video_id: str = Field(..., description="Video ID being processed")
    status: str = Field(..., description="Status of the workflow (started, error)")
    estimated_time: Optional[str] = Field(None, description="Estimated completion time")


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status checks."""
    session_id: str = Field(..., description="Session ID")
    status: str = Field(..., description="Current status (running, completed, failed)")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    progress_percentage: Optional[float] = Field(None, description="Completion percentage (0-100)")
    total_cost: float = Field(..., description="Total cost so far")
    questions_generated: int = Field(..., description="Number of questions generated so far")
    questions_approved: int = Field(..., description="Number of questions approved")
    questions_rejected: int = Field(..., description="Number of questions rejected")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if status is failed")


class VideoResultsResponse(BaseModel):
    """Response model for retrieving video results."""
    video_id: str = Field(..., description="Video ID")
    results: VideoProcessingResult = Field(..., description="Complete processing results")
    results_file_path: str = Field(..., description="Path to the results file")


# ============================================================================
# Error Models
# ============================================================================

class APIError(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")


# ============================================================================
# Status Models
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="API version")
    dependencies: Dict[str, str] = Field(..., description="Status of external dependencies") 