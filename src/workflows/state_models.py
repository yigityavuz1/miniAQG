"""
State models for the AQG LangGraph workflow.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ProcessingStage(str, Enum):
    """Processing stages in the AQG workflow."""
    CONTENT_SPLITTING = "content_splitting"
    SUMMARIZATION = "summarization"
    QUESTION_GENERATION = "question_generation"
    JUDGING = "judging"
    FIXING = "fixing"
    COMPLETED = "completed"
    ERROR = "error"


class QuestionStatus(str, Enum):
    """Status of a question in the workflow."""
    PENDING = "pending"
    APPROVED = "approved" 
    REJECTED = "rejected"
    FAILED = "failed"


class QuestionType(str, Enum):
    """Types of questions that can be generated."""
    MULTIPLE_CHOICE = "multiple-choice"
    TRUE_FALSE = "true-false"
    OPEN_ENDED = "open-ended-short-answer"
    FILL_IN_BLANKS = "fill-in-the-blanks"


class DifficultyLevel(str, Enum):
    """Difficulty levels for questions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ContentSegment(BaseModel):
    """Represents a content segment from the content splitter."""
    topic: str
    subtopics: List[str] = Field(default_factory=list)
    content: str
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    complexity_level: str = "medium"
    summary: Optional[str] = None
    learning_objectives: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)


class ContentSplit(BaseModel):
    """Represents a topic/subtopic split from content."""
    topic_title: str
    subtopic_title: Optional[str] = None
    start_reference: str
    end_reference: str
    estimated_duration_seconds: Optional[int] = None
    content_text: str


class LearningObjective(BaseModel):
    """A learning objective for a content segment."""
    lo_id: str
    lo_text: str
    content_segment_id: str


class SummaryWithLOs(BaseModel):
    """Summary and learning objectives for a content segment."""
    content_segment_id: str
    subtopic_title: str
    summary: str
    learning_objectives: List[LearningObjective]


class QuestionOption(BaseModel):
    """An option for multiple choice questions."""
    option_id: str
    option_text: str


class Question(BaseModel):
    """A generated question."""
    id: str = Field(default_factory=lambda: f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    question_id: str = Field(default_factory=lambda: f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")  # Legacy compatibility
    
    # Question content
    question_type: QuestionType
    question_text: str
    answer_options: Optional[List[str]] = None  # For multiple choice
    options: Optional[List[QuestionOption]] = None  # Legacy compatibility
    correct_answer: Union[str, Dict[str, str]]  # Answer or {"value": "...", "explanation": "..."}
    
    # Learning alignment
    learning_objective_assessed: str
    learning_objective_alignment: str = ""  # For workflow compatibility
    
    # Question metadata
    difficulty_level: DifficultyLevel
    rationale_for_judge: str = ""
    
    # Processing state
    status: QuestionStatus = QuestionStatus.PENDING
    fix_attempts: int = 0
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Judge and fixer tracking
    judge_evaluations: List["JudgeEvaluation"] = Field(default_factory=list)
    fixer_attempts: List["FixerAttempt"] = Field(default_factory=list)


class JudgeEvaluation(BaseModel):
    """Judge evaluation result."""
    question_id: str
    decision: Literal["Approved", "Rejected"]
    pass_status: bool = False  # True if approved, False if rejected
    evaluation_details: List[Dict[str, Any]]
    overall_feedback_summary: str
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)


class FixerAttempt(BaseModel):
    """A fixer attempt to improve a rejected question."""
    attempt_number: int
    original_question: str  # Store as string instead of full Question object
    fixed_question: str
    changes_made: List[str] = Field(default_factory=list)
    reasoning: str = ""
    judge_feedback: "JudgeEvaluation"  # Reference to the evaluation that triggered this fix
    fix_timestamp: datetime = Field(default_factory=datetime.now)


class NodeMetrics(BaseModel):
    """Metrics for a single node execution."""
    node_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    token_usage: Dict[str, int] = Field(default_factory=dict)
    cost: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class SessionMetrics(BaseModel):
    """Overall session metrics."""
    session_id: str
    video_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Content processing metrics
    content_splits_generated: int = 0
    learning_objectives_generated: int = 0
    
    # Question generation metrics
    questions_generated: int = 0
    questions_approved: int = 0
    questions_rejected: int = 0
    questions_failed: int = 0
    
    # Fix loop metrics
    total_fix_attempts: int = 0
    avg_fix_iterations_per_question: float = 0.0
    max_fix_iterations_reached: int = 0
    
    # Performance metrics
    node_metrics: List[NodeMetrics] = Field(default_factory=list)
    total_tokens_used: Dict[str, int] = Field(default_factory=dict)
    total_cost: float = 0.0
    
    # Quality metrics
    approval_rate: float = 0.0
    avg_processing_time_per_question: float = 0.0


class AQGState(BaseModel):
    """Main state for the AQG workflow using LangGraph."""
    
    # Session identification
    session_id: str = Field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    video_id: str
    video_title: str
    
    # Input data
    transcript: str = ""  # Full transcript text
    transcript_content: Dict[str, Any] = Field(default_factory=dict)  # Original JSON data
    
    # Current processing stage
    current_stage: ProcessingStage = ProcessingStage.CONTENT_SPLITTING
    
    # Content processing results
    content_segments: List[ContentSegment] = Field(default_factory=list)
    
    # Processing stages results (legacy compatibility)
    content_splits: List[ContentSplit] = Field(default_factory=list)
    summaries_and_los: List[SummaryWithLOs] = Field(default_factory=list)
    
    # Question management
    questions: List[Question] = Field(default_factory=list)  # Main questions list
    generated_questions: List[Question] = Field(default_factory=list)  # Legacy compatibility
    current_question: Optional[Question] = None
    current_lo_being_processed: Optional[str] = None
    
    # Judge/Fixer loop tracking
    judge_iteration: int = 0
    evaluation_log: List[JudgeEvaluation] = Field(default_factory=list)
    fixer_attempts: List[FixerAttempt] = Field(default_factory=list)
    current_fix_attempts: int = 0
    max_fix_attempts: int = 5
    
    # Workflow control
    current_node: Optional[str] = None
    processing_start_time: datetime = Field(default_factory=datetime.now)
    workflow_completed: bool = False
    
    # Stage outputs for tracking
    stage_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Cost tracking
    total_cost: float = 0.0
    
    # Final result
    final_result: Optional[VideoProcessingResult] = None
    
    # Prompt versioning for traceability
    prompt_versions: Dict[str, str] = Field(default_factory=dict)
    
    # Metrics and monitoring
    session_metrics: SessionMetrics = Field(default_factory=lambda: SessionMetrics(
        session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        video_id=""
    ))
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize session metrics with correct video_id after model creation."""
        if self.session_metrics.video_id == "":
            self.session_metrics.video_id = self.video_id
            self.session_metrics.session_id = self.session_id
    
    def add_content_split(self, split: ContentSplit) -> None:
        """Add a content split and update metrics."""
        self.content_splits.append(split)
        self.session_metrics.content_splits_generated += 1
    
    def add_summary_with_los(self, summary_los: SummaryWithLOs) -> None:
        """Add summary with learning objectives and update metrics."""
        self.summaries_and_los.append(summary_los)
        self.session_metrics.learning_objectives_generated += len(summary_los.learning_objectives)
    
    def add_generated_question(self, question: Question) -> None:
        """Add a generated question and update metrics."""
        self.generated_questions.append(question)
        self.session_metrics.questions_generated += 1
    
    def add_judge_evaluation(self, evaluation: JudgeEvaluation) -> None:
        """Add judge evaluation and update metrics."""
        self.evaluation_log.append(evaluation)
        
        if evaluation.decision == "Approved":
            self.session_metrics.questions_approved += 1
        else:
            self.session_metrics.questions_rejected += 1
    
    def add_fixer_attempt(self, attempt: FixerAttempt) -> None:
        """Add fixer attempt and update metrics."""
        self.fixer_attempts.append(attempt)
        self.session_metrics.total_fix_attempts += 1
        self.current_fix_attempts += 1
    
    def reset_fix_attempts(self) -> None:
        """Reset fix attempts counter for new question."""
        self.current_fix_attempts = 0
    
    def mark_question_failed(self) -> None:
        """Mark current question as failed after max attempts."""
        if self.current_question:
            self.current_question.status = QuestionStatus.FAILED
            self.session_metrics.questions_failed += 1
            self.session_metrics.max_fix_iterations_reached += 1
    
    def add_node_metrics(self, metrics: NodeMetrics) -> None:
        """Add node execution metrics."""
        self.session_metrics.node_metrics.append(metrics)
        
        # Update total tokens and cost
        for provider, tokens in metrics.token_usage.items():
            self.session_metrics.total_tokens_used[provider] = (
                self.session_metrics.total_tokens_used.get(provider, 0) + tokens
            )
        
        if metrics.cost:
            self.session_metrics.total_cost += metrics.cost
    
    def finalize_session_metrics(self) -> None:
        """Finalize session metrics at the end of processing."""
        self.session_metrics.end_time = datetime.now()
        
        if self.session_metrics.start_time and self.session_metrics.end_time:
            self.session_metrics.total_duration_seconds = (
                self.session_metrics.end_time - self.session_metrics.start_time
            ).total_seconds()
        
        # Calculate approval rate
        total_evaluated = self.session_metrics.questions_approved + self.session_metrics.questions_rejected
        if total_evaluated > 0:
            self.session_metrics.approval_rate = self.session_metrics.questions_approved / total_evaluated
        
        # Calculate average fix iterations per question
        if self.session_metrics.questions_generated > 0:
            self.session_metrics.avg_fix_iterations_per_question = (
                self.session_metrics.total_fix_attempts / self.session_metrics.questions_generated
            )
        
        # Calculate average processing time per question
        if (self.session_metrics.questions_generated > 0 and 
            self.session_metrics.total_duration_seconds):
            self.session_metrics.avg_processing_time_per_question = (
                self.session_metrics.total_duration_seconds / self.session_metrics.questions_generated
            )
    
    def get_current_question_for_processing(self) -> Optional[Question]:
        """Get the current question being processed (for judge/fixer loop)."""
        return self.current_question
    
    def update_current_question(self, question: Question) -> None:
        """Update the current question being processed."""
        self.current_question = question
        
        # Find and update in the generated questions list
        for i, q in enumerate(self.generated_questions):
            if q.question_id == question.question_id:
                self.generated_questions[i] = question
                break


class VideoProcessingResult(BaseModel):
    """Final result of processing a video through the AQG workflow."""
    video_id: str
    video_title: str
    session_id: str
    processing_start_time: datetime
    processing_end_time: datetime
    processing_time: float  # seconds
    
    # Content processing results
    content_segments_count: int
    total_learning_objectives: int
    
    # Question generation results
    total_questions_generated: int
    approved_questions: int
    rejected_questions: int
    failed_questions: int
    
    # Judge/fixer loop results
    judge_iterations: int
    total_fix_attempts: int
    questions_requiring_fixes: int
    
    # Cost and performance metrics
    total_cost: float
    cost_per_approved_question: float
    avg_processing_time_per_question: float
    
    # Quality metrics
    approval_rate: float
    avg_fix_iterations_per_question: float
    
    # Generated content
    final_questions: List[Question]
    
    # Metadata
    prompt_versions_used: Dict[str, str] = Field(default_factory=dict)
    model_providers_used: Dict[str, str] = Field(default_factory=dict) 