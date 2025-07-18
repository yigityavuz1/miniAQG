"""
AQG Workflow Implementation using LangGraph.
Updated with latest LangGraph best practices (June 2025).
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .state_models import (
    AQGState, 
    Question, 
    JudgeEvaluation, 
    FixerAttempt,
    ContentSegment,
    ProcessingStage,
    VideoProcessingResult
)
from ..core.llm_client import LLMClient
from ..core.prompt_loader import PromptLoader
from ..core.logger import AQGLogger


class AQGWorkflow:
    """
    Main AQG workflow orchestrator using LangGraph.
    Implements the complete pipeline from transcript to validated questions.
    """
    
    def __init__(self, llm_client: LLMClient, logger: Optional[AQGLogger] = None):
        self.llm_client = llm_client
        self.logger = logger or AQGLogger()
        self.prompt_loader = PromptLoader()
        
        # Initialize memory for state persistence (in-memory only)
        self.memory = MemorySaver()
        
        # Build the workflow graph
        self.graph = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with all nodes and edges."""
        
        # Initialize StateGraph with our state model
        workflow = StateGraph(AQGState)
        
        # Add all workflow nodes
        workflow.add_node("content_splitter", self._content_splitter_node)
        workflow.add_node("summarizer", self._summarizer_node)
        workflow.add_node("question_generator", self._question_generator_node)
        workflow.add_node("judge", self._judge_node)
        workflow.add_node("fixer", self._fixer_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        
        # Define workflow edges
        workflow.add_edge(START, "content_splitter")
        workflow.add_edge("content_splitter", "summarizer")
        workflow.add_edge("summarizer", "question_generator")
        workflow.add_edge("question_generator", "judge")
        
        # Conditional edge from judge - decides whether to fix or finalize
        workflow.add_conditional_edges(
            "judge",
            self._judge_decision_router,
            {
                "fix_needed": "fixer",
                "approved": "finalize_results",
                "max_attempts_reached": "finalize_results"
            }
        )
        
        # Edge from fixer back to judge for re-evaluation
        workflow.add_edge("fixer", "judge")
        workflow.add_edge("finalize_results", END)
        
        # Compile the workflow with memory checkpointing
        return workflow.compile(checkpointer=self.memory)
    
    async def _content_splitter_node(self, state: AQGState) -> Dict[str, Any]:
        """
        Node 1: Split transcript content into hierarchical segments.
        """
        self.logger.info("Starting content splitter node")
        
        # Update stage
        state.current_stage = ProcessingStage.CONTENT_SPLITTING
        
        # Load prompt
        prompt_template = self.prompt_loader.load_prompt("content_splitter", "v1")
        
        # Format prompt with actual variables from the prompt template
        prompt = prompt_template.replace("{video_title}", state.video_title)
        prompt = prompt.replace("{video_transcript}", state.transcript)
        
        # Call LLM
        response = await self.llm_client.call_model(
            prompt=prompt,
            model="gpt-4o",
            system_instruction="You are an expert content analyzer specializing in educational video transcripts. Return valid JSON only."
        )
        
        # Parse response into content segments
        try:
            content_data = json.loads(response.content)
            segments = []
            
            # Parse the structure from the prompt template
            for topic in content_data.get("topics", []):
                topic_title = topic.get("topic_title", "Unknown Topic")
                
                # Create segments for each subtopic
                for subtopic in topic.get("subtopics", []):
                    # Extract the actual content from transcript using references
                    start_ref = subtopic.get("start_reference", "")
                    end_ref = subtopic.get("end_reference", "")
                    
                    # For now, use the full transcript content (we can optimize this later)
                    segment_content = state.transcript
                    
                    segment = ContentSegment(
                        topic=topic_title,
                        subtopics=[subtopic.get("subtopic_title", "")],
                        content=segment_content,
                        timestamp_start=None,  # Will be parsed from references later
                        timestamp_end=None,
                        complexity_level="medium"
                    )
                    segments.append(segment)
            
            # If no segments were created, create a fallback
            if not segments:
                segments = [
                    ContentSegment(
                        topic="Main Content",
                        subtopics=[],
                        content=state.transcript,
                        complexity_level="medium"
                    )
                ]
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse content splitter response as JSON: {e}")
            # Log the actual response for debugging
            self.logger.error(f"Response content: {response.content[:500]}...")
            
            # Fallback: create single segment
            segments = [
                ContentSegment(
                    topic="Main Content",
                    subtopics=[],
                    content=state.transcript,
                    complexity_level="medium"
                )
            ]
        
        # Update state
        state.content_segments = segments
        state.stage_outputs["content_splitting"] = {
            "segments_count": len(segments),
            "processing_time": response.response_time,
            "cost": response.usage.cost_usd
        }
        
        self.logger.info(f"Content splitter completed: {len(segments)} segments created")
        
        return {
            "content_segments": segments,
            "current_stage": ProcessingStage.CONTENT_SPLITTING,
            "stage_outputs": state.stage_outputs,
            "total_cost": state.total_cost + response.usage.cost_usd
        }
    
    async def _summarizer_node(self, state: AQGState) -> Dict[str, Any]:
        """
        Node 2: Create summaries and learning objectives from content segments.
        """
        self.logger.info("Starting summarizer node")
        
        state.current_stage = ProcessingStage.SUMMARIZATION
        
        # Load prompt
        prompt_template = self.prompt_loader.load_prompt("summarizer", "v1")
        
        # Process each segment
        for i, segment in enumerate(state.content_segments):
            self.logger.info(f"Summarizing segment {i+1}/{len(state.content_segments)}")
            
            # Format prompt
            prompt = prompt_template.replace("{topic}", segment.topic)
            prompt = prompt.replace("{subtopics}", ", ".join(segment.subtopics))
            prompt = prompt.replace("{content}", segment.content)
            
            # Call LLM
            response = await self.llm_client.call_model(
                prompt=prompt,
                model="gpt-4o",
                system_instruction="You are an expert educational content curator specializing in creating SMART learning objectives."
            )
            
            # Parse response to update segment
            try:
                summary_data = json.loads(response.content)
                segment.summary = summary_data.get("summary", "")
                
                # Handle learning objectives that might be in dict format
                raw_objectives = summary_data.get("learning_objectives", [])
                if raw_objectives and isinstance(raw_objectives[0], dict):
                    # Extract text from dict format: {"lo_id": "LO1", "lo_text": "..."}
                    segment.learning_objectives = [obj.get("lo_text", str(obj)) for obj in raw_objectives]
                else:
                    # Already in string format
                    segment.learning_objectives = raw_objectives
                
                segment.key_concepts = summary_data.get("key_concepts", [])
                
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse summarizer response for segment {i+1}")
                segment.summary = response.content[:500] + "..." if len(response.content) > 500 else response.content
                segment.learning_objectives = ["Understand the main concepts from this content segment"]
        
        # Update stage outputs
        state.stage_outputs["summarization"] = {
            "segments_processed": len(state.content_segments),
            "total_learning_objectives": sum(len(seg.learning_objectives) for seg in state.content_segments),
            "processing_time": 0.0,  # Will be calculated from actual LLM calls
            "cost": 0.0  # Will be calculated from actual LLM calls
        }
        
        self.logger.info("Summarizer node completed")
        
        return {
            "content_segments": state.content_segments,
            "current_stage": ProcessingStage.SUMMARIZATION,
            "stage_outputs": state.stage_outputs
        }
    
    async def _question_generator_node(self, state: AQGState) -> Dict[str, Any]:
        """
        Node 3: Generate assessment questions from learning objectives.
        """
        self.logger.info("Starting question generator node")
        
        state.current_stage = ProcessingStage.QUESTION_GENERATION
        
        # Load prompt
        prompt_template = self.prompt_loader.load_prompt("question_generator", "v1")
        
        questions = []
        
        # Generate questions for each content segment
        for i, segment in enumerate(state.content_segments):
            self.logger.info(f"Generating questions for segment {i+1}/{len(state.content_segments)}")
            
            # Ensure we have strings for prompt formatting
            learning_objectives_text = "\n".join(segment.learning_objectives) if segment.learning_objectives else "General understanding of the content"
            summary_text = segment.summary if hasattr(segment, 'summary') and segment.summary else "Content summary not available"
            key_concepts_text = ", ".join(segment.key_concepts) if hasattr(segment, 'key_concepts') and segment.key_concepts else "Key concepts not available"
            
            # Format prompt
            prompt = prompt_template.replace("{topic}", segment.topic)
            prompt = prompt.replace("{learning_objectives}", learning_objectives_text)
            prompt = prompt.replace("{summary}", summary_text)
            prompt = prompt.replace("{key_concepts}", key_concepts_text)
            
            # Call LLM
            response = await self.llm_client.call_model(
                prompt=prompt,
                model="gpt-4o",
                system_instruction="You are an expert educational assessment designer specializing in creating diverse, high-quality questions."
            )
            
            # Parse response
            try:
                questions_data = json.loads(response.content)
                self.logger.info(f"Question generator response for segment {i+1}: {questions_data}")
                
                # Handle both single question and questions array formats
                if "questions" in questions_data:
                    # Array format: {"questions": [...]}
                    questions_list = questions_data["questions"]
                else:
                    # Single question format: {...}
                    questions_list = [questions_data]
                
                for q_data in questions_list:
                    # Handle different answer option formats
                    answer_options = []
                    if "options" in q_data:
                        # Format: [{"option_id": "A", "option_text": "..."}]
                        answer_options = [opt.get("option_text", str(opt)) for opt in q_data["options"]]
                    elif "answer_options" in q_data:
                        # Format: ["A", "B", "C", "D"]
                        answer_options = q_data["answer_options"]
                    
                    # Handle correct answer format
                    correct_answer = ""
                    if "correct_answer" in q_data:
                        ca = q_data["correct_answer"]
                        if isinstance(ca, dict):
                            correct_answer = ca.get("value", "") + ": " + ca.get("explanation", "")
                        else:
                            correct_answer = str(ca)
                    
                    question = Question(
                        id=f"q_{len(questions) + 1}",
                        question_type=q_data.get("question_type", "multiple-choice"),
                        question_text=q_data.get("question_text", ""),
                        correct_answer=correct_answer,
                        answer_options=answer_options,
                        learning_objective_assessed=q_data.get("learning_objective_assessed", ""),
                        learning_objective_alignment=q_data.get("learning_objective_assessed", ""),
                        difficulty_level=q_data.get("difficulty_level", "medium"),
                        rationale_for_judge=q_data.get("rationale_for_judge", "")
                    )
                    questions.append(question)
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse question generator response for segment {i+1}: {e}")
                self.logger.error(f"Raw response: {response.content[:500]}...")
        
        # Update state
        state.questions = questions
        state.stage_outputs["question_generation"] = {
            "questions_generated": len(questions),
            "segments_processed": len(state.content_segments),
            "questions_per_segment": len(questions) / max(1, len(state.content_segments))
        }
        
        self.logger.info(f"Question generator completed: {len(questions)} questions generated")
        
        return {
            "questions": questions,
            "current_stage": ProcessingStage.QUESTION_GENERATION,
            "stage_outputs": state.stage_outputs
        }
    
    async def _judge_node(self, state: AQGState) -> Dict[str, Any]:
        """
        Node 4: Evaluate questions against quality criteria.
        """
        self.logger.info("Starting judge node")
        
        state.current_stage = ProcessingStage.JUDGING
        
        # Load prompt
        prompt_template = self.prompt_loader.load_prompt("judge", "v1")
        
        # Process questions in batches for efficiency
        batch_size = 5
        
        for i in range(0, len(state.questions), batch_size):
            batch = state.questions[i:i + batch_size]
            self.logger.info(f"Judging batch {i//batch_size + 1}: questions {i+1}-{min(i+batch_size, len(state.questions))}")
            
            # Format questions for judgment
            questions_text = "\n\n".join([
                f"Question {q.id}:\n"
                f"Type: {q.question_type}\n"
                f"Text: {q.question_text}\n"
                f"Answer: {q.correct_answer}\n"
                f"Options: {q.answer_options}\n"
                f"LO Alignment: {q.learning_objective_alignment}"
                for q in batch
            ])
            
            # Format prompt
            prompt = prompt_template.replace("{questions}", questions_text)
            
            # Call LLM
            response = await self.llm_client.call_model(
                prompt=prompt,
                model="gpt-4o",
                system_instruction="You are a rigorous educational assessment expert with expertise in question quality evaluation."
            )
            
            # Parse response
            try:
                evaluations_data = json.loads(response.content)
                self.logger.info(f"Judge response for batch {i//batch_size + 1}: {evaluations_data}")
                
                # Handle both single evaluation and evaluations array formats
                if "evaluations" in evaluations_data:
                    # Array format: {"evaluations": [...]}
                    evaluations_list = evaluations_data["evaluations"]
                else:
                    # Single evaluation format: {...}
                    evaluations_list = [evaluations_data]
                
                for idx, eval_data in enumerate(evaluations_list):
                    # Match by index since question_id from judge may not match
                    if idx < len(batch):
                        question = batch[idx]
                        
                        # Set pass_status based on decision field
                        decision = eval_data.get("decision", "Rejected")
                        pass_status = (decision.lower() == "approved")
                        
                        evaluation = JudgeEvaluation(
                            question_id=question.id,  # Use the actual question ID
                            decision=decision,
                            pass_status=pass_status,
                            evaluation_details=eval_data.get("evaluation_details", []),
                            overall_feedback_summary=eval_data.get("overall_feedback_summary", "")
                        )
                        
                        self.logger.info(f"Created evaluation for {question.id}: decision='{decision}', pass_status={pass_status}")
                        question.judge_evaluations.append(evaluation)
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse judge response for batch {i//batch_size + 1}: {e}")
                self.logger.error(f"Raw response: {response.content[:500]}...")
        
        # Update judge iteration
        state.judge_iteration += 1
        
        # Calculate summary statistics
        approved_questions = [q for q in state.questions if q.judge_evaluations and q.judge_evaluations[-1].pass_status]
        rejected_questions = [q for q in state.questions if q.judge_evaluations and not q.judge_evaluations[-1].pass_status]
        
        state.stage_outputs["judging"] = {
            "total_questions_evaluated": len(state.questions),
            "approved_questions": len(approved_questions),
            "rejected_questions": len(rejected_questions),
            "approval_rate": len(approved_questions) / max(1, len(state.questions)),
            "judge_iteration": state.judge_iteration
        }
        
        self.logger.info(f"Judge node completed: {len(approved_questions)}/{len(state.questions)} questions approved")
        
        return {
            "questions": state.questions,
            "judge_iteration": state.judge_iteration,
            "current_stage": ProcessingStage.JUDGING,
            "stage_outputs": state.stage_outputs
        }
    
    def _judge_decision_router(self, state: AQGState) -> Literal["fix_needed", "approved", "max_attempts_reached"]:
        """
        Conditional edge function to determine next step after judging.
        """
        # Check if we've reached maximum fix attempts
        if state.judge_iteration >= 5:  # Max 5 iterations
            self.logger.warning("Maximum judge iterations reached")
            return "max_attempts_reached"
        
        # Debug: Log all question evaluation statuses
        for i, q in enumerate(state.questions):
            if q.judge_evaluations:
                latest_eval = q.judge_evaluations[-1]
                self.logger.info(f"Question {q.id}: decision='{latest_eval.decision}', pass_status={latest_eval.pass_status}")
            else:
                self.logger.info(f"Question {q.id}: No evaluations")
        
        # Check if there are rejected questions that need fixing
        rejected_questions = [
            q for q in state.questions 
            if q.judge_evaluations and not q.judge_evaluations[-1].pass_status
        ]
        
        if rejected_questions:
            self.logger.info(f"{len(rejected_questions)} questions need fixing")
            return "fix_needed"
        else:
            self.logger.info("All questions approved")
            return "approved"
    
    async def _fixer_node(self, state: AQGState) -> Dict[str, Any]:
        """
        Node 5: Fix rejected questions based on judge feedback.
        """
        self.logger.info("Starting fixer node")
        
        state.current_stage = ProcessingStage.FIXING
        
        # Load prompt
        prompt_template = self.prompt_loader.load_prompt("fixer", "v1")
        
        # Find questions that need fixing
        rejected_questions = [
            q for q in state.questions 
            if q.judge_evaluations and not q.judge_evaluations[-1].pass_status
        ]
        
        for question in rejected_questions:
            self.logger.info(f"Fixing question {question.id}")
            
            # Get latest evaluation
            latest_eval = question.judge_evaluations[-1]
            
            # Format prompt
            prompt = prompt_template.replace("{original_question}", question.question_text)
            prompt = prompt.replace("{question_type}", str(question.question_type))
            prompt = prompt.replace("{correct_answer}", str(question.correct_answer))
            prompt = prompt.replace("{answer_options}", str(question.answer_options))
            prompt = prompt.replace("{judge_feedback}", latest_eval.overall_feedback_summary)
            prompt = prompt.replace("{suggested_improvements}", latest_eval.overall_feedback_summary)
            prompt = prompt.replace("{criteria_scores}", str(latest_eval.evaluation_details))
            
            # Call LLM
            response = await self.llm_client.call_model(
                prompt=prompt,
                model="gpt-4o",
                system_instruction="You are an expert educational question designer specializing in improving question quality based on detailed feedback."
            )
            
            # Parse response
            try:
                fix_data = json.loads(response.content)
                
                # Create fixer attempt record
                fixer_attempt = FixerAttempt(
                    attempt_number=len(question.fixer_attempts) + 1,
                    original_question=question.question_text,
                    fixed_question=fix_data.get("fixed_question", ""),
                    changes_made=fix_data.get("changes_made", []),
                    reasoning=fix_data.get("reasoning", ""),
                    judge_feedback=latest_eval
                )
                
                question.fixer_attempts.append(fixer_attempt)
                
                # Update question with fixed version
                question.question_text = fix_data.get("fixed_question", question.question_text)
                question.correct_answer = fix_data.get("fixed_answer", question.correct_answer)
                question.answer_options = fix_data.get("fixed_options", question.answer_options)
                question.rationale_for_judge = fix_data.get("improved_rationale", question.rationale_for_judge)
                
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse fixer response for question {question.id}")
        
        # Update stage outputs
        state.stage_outputs["fixing"] = {
            "questions_fixed": len(rejected_questions),
            "total_fix_attempts": sum(len(q.fixer_attempts) for q in state.questions),
            "current_iteration": state.judge_iteration
        }
        
        self.logger.info(f"Fixer node completed: {len(rejected_questions)} questions processed")
        
        return {
            "questions": state.questions,
            "current_stage": ProcessingStage.FIXING,
            "stage_outputs": state.stage_outputs
        }
    
    async def _finalize_results_node(self, state: AQGState) -> Dict[str, Any]:
        """
        Node 6: Finalize results and prepare output.
        """
        self.logger.info("Starting finalize results node")
        
        state.current_stage = ProcessingStage.COMPLETED
        
        # Calculate final statistics
        approved_questions = [
            q for q in state.questions 
            if q.judge_evaluations and q.judge_evaluations[-1].pass_status
        ]
        
        rejected_questions = [
            q for q in state.questions 
            if q.judge_evaluations and not q.judge_evaluations[-1].pass_status
        ]
        
        # Create final result
        processing_end_time = datetime.now()
        processing_time = (processing_end_time - state.processing_start_time).total_seconds()
        
        result = VideoProcessingResult(
            video_id=state.video_id,
            video_title=state.video_title,
            session_id=state.session_id,
            processing_start_time=state.processing_start_time,
            processing_end_time=processing_end_time,
            processing_time=processing_time,
            
            # Content processing results
            content_segments_count=len(state.content_segments),
            total_learning_objectives=sum(len(seg.learning_objectives) for seg in state.content_segments),
            
            # Question generation results
            total_questions_generated=len(state.questions),
            approved_questions=len(approved_questions),
            rejected_questions=len(rejected_questions),
            failed_questions=0,  # We don't have failed questions yet
            
            # Judge/fixer loop results
            judge_iterations=state.judge_iteration,
            total_fix_attempts=sum(len(q.fixer_attempts) for q in state.questions),
            questions_requiring_fixes=len([q for q in state.questions if q.fixer_attempts]),
            
            # Cost and performance metrics
            total_cost=state.total_cost,
            cost_per_approved_question=state.total_cost / max(1, len(approved_questions)),
            avg_processing_time_per_question=processing_time / max(1, len(state.questions)),
            
            # Quality metrics
            approval_rate=len(approved_questions) / max(1, len(state.questions)),
            avg_fix_iterations_per_question=sum(len(q.fixer_attempts) for q in state.questions) / max(1, len(state.questions)),
            
            # Generated content
            final_questions=approved_questions,
            
            # Metadata
            prompt_versions_used=state.prompt_versions,
            model_providers_used={"all": "openai"}  # Since we're using gpt-4o for all
        )
        
        state.final_result = result
        
        # Log final summary
        self.logger.info(
            f"AQG workflow completed for {state.video_title}: "
            f"{len(approved_questions)}/{len(state.questions)} questions approved, "
            f"cost: ${state.total_cost:.4f}"
        )
        
        return {
            "current_stage": ProcessingStage.COMPLETED,
            "final_result": result,
            "completed": True
        }
    
    async def process_video(
        self,
        video_id: str,
        video_title: str,
        transcript: str,
        config: Optional[Dict[str, Any]] = None
    ) -> VideoProcessingResult:
        """
        Process a single video through the complete AQG workflow.
        
        Args:
            video_id: Unique identifier for the video
            video_title: Title of the video
            transcript: Video transcript text
            config: Optional configuration overrides
            
        Returns:
            VideoProcessingResult with all processing details
        """
        self.logger.info(f"Starting AQG workflow for video: {video_title}")
        
        # Create initial state
        initial_state = AQGState(
            video_id=video_id,
            video_title=video_title,
            transcript=transcript,
            current_stage=ProcessingStage.CONTENT_SPLITTING,
            questions=[],
            content_segments=[],
            judge_iteration=0,
            total_cost=0.0,
            stage_outputs={}
        )
        
        # Create a unique thread ID for this processing session
        thread_id = f"video_{video_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Execute the workflow
            final_state = await self.graph.ainvoke(
                initial_state.model_dump(),
                config={"configurable": {"thread_id": thread_id}}
            )
            
            # Return the final result
            return final_state["final_result"]
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed for {video_title}: {e}")
            
            # Return error result
            error_time = datetime.now()
            return VideoProcessingResult(
                video_id=video_id,
                video_title=video_title,
                session_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                processing_start_time=initial_state.processing_start_time,
                processing_end_time=error_time,
                processing_time=(error_time - initial_state.processing_start_time).total_seconds(),
                
                # Content processing results
                content_segments_count=0,
                total_learning_objectives=0,
                
                # Question generation results
                total_questions_generated=0,
                approved_questions=0,
                rejected_questions=0,
                failed_questions=0,
                
                # Judge/fixer loop results
                judge_iterations=0,
                total_fix_attempts=0,
                questions_requiring_fixes=0,
                
                # Cost and performance metrics
                total_cost=0.0,
                cost_per_approved_question=0.0,
                avg_processing_time_per_question=0.0,
                
                # Quality metrics
                approval_rate=0.0,
                avg_fix_iterations_per_question=0.0,
                
                # Generated content
                final_questions=[],
                
                # Metadata
                prompt_versions_used={},
                model_providers_used={}
            )
    
    def get_workflow_graph(self) -> str:
        """Get Mermaid diagram representation of the workflow."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            self.logger.error(f"Failed to generate workflow diagram: {e}")
            return "graph TD; A[Error generating diagram]" 