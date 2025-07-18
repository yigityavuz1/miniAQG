"""
Comprehensive logging system for the AQG workflow.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..workflows.state_models import AQGState, NodeMetrics, Question, JudgeEvaluation


class AQGLogger:
    """Comprehensive logging for all workflow steps."""
    
    def __init__(self, session_id: Optional[str] = None):
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        self.setup_loggers()
        self.session_start_time = datetime.now()
        
    def setup_loggers(self):
        """Set up console and file loggers."""
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console logger
        self.console_logger = logging.getLogger(f"aqg_console_{self.session_id}")
        self.console_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.console_logger.handlers[:]:
            self.console_logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)
        
        # File logger with timestamp
        log_file = self.logs_dir / f"{self.session_id}.log"
        self.file_logger = logging.getLogger(f"aqg_file_{self.session_id}")
        self.file_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.console_logger.propagate = False
        self.file_logger.propagate = False
        
        self.info("AQG Logger initialized", {"session_id": self.session_id})
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log info level message."""
        self._log(logging.INFO, message, data)
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log debug level message."""
        self._log(logging.DEBUG, message, data)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log warning level message."""
        self._log(logging.WARNING, message, data)
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log error level message."""
        self._log(logging.ERROR, message, data)
    
    def _log(self, level: int, message: str, data: Optional[Dict[str, Any]] = None):
        """Internal logging method."""
        log_message = message
        if data:
            log_message += f" | Data: {json.dumps(data, default=str, ensure_ascii=False)}"
        
        self.console_logger.log(level, log_message)
        self.file_logger.log(level, log_message)
    
    def log_workflow_start(self, state: AQGState):
        """Log when the workflow starts."""
        self.info("=== AQG Workflow Started ===", {
            "session_id": state.session_id,
            "video_id": state.video_id,
            "video_title": state.video_title,
            "prompt_versions": state.prompt_versions,
            "max_fix_attempts": state.max_fix_attempts
        })
    
    def log_workflow_complete(self, state: AQGState):
        """Log when the workflow completes."""
        metrics = state.session_metrics
        self.info("=== AQG Workflow Completed ===", {
            "session_id": state.session_id,
            "duration_seconds": metrics.total_duration_seconds,
            "questions_generated": metrics.questions_generated,
            "questions_approved": metrics.questions_approved,
            "questions_rejected": metrics.questions_rejected,
            "questions_failed": metrics.questions_failed,
            "approval_rate": f"{metrics.approval_rate:.2%}",
            "total_cost": f"${metrics.total_cost:.4f}",
            "avg_cost_per_question": f"${metrics.total_cost / max(metrics.questions_generated, 1):.4f}"
        })
    
    def log_node_start(self, node_name: str, input_data: Dict[str, Any]):
        """Log when a LangGraph node starts processing."""
        self.info(f"Node {node_name} started", {
            "node": node_name,
            "input_keys": list(input_data.keys()) if input_data else []
        })
    
    def log_node_complete(
        self, 
        node_name: str, 
        output_data: Dict[str, Any], 
        metrics: NodeMetrics
    ):
        """Log when a LangGraph node completes."""
        self.info(f"Node {node_name} completed", {
            "node": node_name,
            "duration_seconds": metrics.duration_seconds,
            "success": metrics.success,
            "token_usage": metrics.token_usage,
            "cost": f"${metrics.cost:.4f}" if metrics.cost else "N/A",
            "output_keys": list(output_data.keys()) if output_data else []
        })
        
        if not metrics.success and metrics.error_message:
            self.error(f"Node {node_name} failed", {"error": metrics.error_message})
    
    def log_content_split(self, video_id: str, splits_count: int):
        """Log content splitting results."""
        self.info("Content split completed", {
            "video_id": video_id,
            "splits_generated": splits_count
        })
    
    def log_learning_objectives(self, segment_id: str, los_count: int):
        """Log learning objectives generation."""
        self.info("Learning objectives generated", {
            "segment_id": segment_id,
            "objectives_count": los_count
        })
    
    def log_question_generation(self, question: Question):
        """Log when a question is generated."""
        self.info("Question generated", {
            "question_id": question.question_id,
            "question_type": question.question_type.value,
            "difficulty": question.difficulty_level.value,
            "lo_assessed": question.learning_objective_assessed
        })
        
        self.debug("Question details", {
            "question_id": question.question_id,
            "question_text": question.question_text[:200] + "..." if len(question.question_text) > 200 else question.question_text,
            "correct_answer": question.correct_answer.get("value", "N/A")
        })
    
    def log_judge_decision(self, question: Question, evaluation: JudgeEvaluation):
        """Detailed logging of judge decisions for analysis."""
        decision_data = {
            "question_id": question.question_id,
            "decision": evaluation.decision,
            "criteria_passed": sum(1 for detail in evaluation.evaluation_details if detail.get("status") == "Pass"),
            "criteria_failed": sum(1 for detail in evaluation.evaluation_details if detail.get("status") == "Fail"),
            "overall_feedback": evaluation.overall_feedback_summary
        }
        
        if evaluation.decision == "Approved":
            self.info("Question approved by judge", decision_data)
        else:
            self.warning("Question rejected by judge", decision_data)
            
            # Log detailed feedback for rejected questions
            self.debug("Judge rejection details", {
                "question_id": question.question_id,
                "evaluation_details": evaluation.evaluation_details
            })
    
    def log_fixer_attempt(self, attempt_number: int, question_id: str, success: bool):
        """Log fixer attempts."""
        if success:
            self.info("Question fixed successfully", {
                "question_id": question_id,
                "attempt_number": attempt_number
            })
        else:
            self.warning("Question fix attempt failed", {
                "question_id": question_id,
                "attempt_number": attempt_number
            })
    
    def log_question_failed(self, question_id: str, max_attempts: int):
        """Log when a question fails after max attempts."""
        self.error("Question failed after max attempts", {
            "question_id": question_id,
            "max_attempts": max_attempts
        })
    
    def log_cost_alert(self, alert_message: str):
        """Log cost alerts."""
        self.warning("Cost Alert", {"message": alert_message})
    
    def log_llm_call(
        self, 
        agent_name: str, 
        model: str, 
        tokens_used: int, 
        cost: float, 
        duration: float
    ):
        """Log LLM API calls for cost tracking."""
        self.debug("LLM call made", {
            "agent": agent_name,
            "model": model,
            "tokens": tokens_used,
            "cost": f"${cost:.4f}",
            "duration": f"{duration:.2f}s"
        })
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log errors with additional context."""
        self.error(f"Error occurred: {str(error)}", {
            "error_type": type(error).__name__,
            "context": context
        })
    
    def export_metrics_summary(self, state: AQGState) -> str:
        """Export session metrics to JSON file."""
        metrics_file = self.logs_dir / f"metrics_{self.session_id}.json"
        
        # Prepare metrics data
        metrics_data = {
            "session_info": {
                "session_id": state.session_id,
                "video_id": state.video_id,
                "video_title": state.video_title,
                "start_time": state.processing_start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            },
            "workflow_metrics": state.session_metrics.model_dump(),
            "prompt_versions": state.prompt_versions,
            "questions_summary": [
                {
                    "question_id": q.question_id,
                    "status": q.status.value,
                    "question_type": q.question_type.value,
                    "difficulty": q.difficulty_level.value,
                    "fix_attempts": q.fix_attempts,
                    "generation_time": q.generation_timestamp.isoformat()
                }
                for q in state.generated_questions
            ],
            "evaluations_summary": [
                {
                    "question_id": eval.question_id,
                    "decision": eval.decision,
                    "evaluation_time": eval.evaluation_timestamp.isoformat(),
                    "feedback_summary": eval.overall_feedback_summary
                }
                for eval in state.evaluation_log
            ]
        }
        
        try:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            self.info("Metrics exported", {"file": str(metrics_file)})
            return str(metrics_file)
            
        except Exception as e:
            self.error("Failed to export metrics", {"error": str(e)})
            return ""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat(),
            "duration": (datetime.now() - self.session_start_time).total_seconds(),
            "log_file": str(self.logs_dir / f"{self.session_id}.log")
        }


# Global logger instance that can be imported
_global_logger: Optional[AQGLogger] = None

def get_logger(session_id: Optional[str] = None) -> AQGLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None or (session_id and session_id != _global_logger.session_id):
        _global_logger = AQGLogger(session_id)
    return _global_logger

def set_logger(logger: AQGLogger) -> None:
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger 