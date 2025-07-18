# AQG Project: Current State Summary

This document summarizes the initial goals, what has been accomplished, and the current state of the Mini AQG (Automated Question Generation) project.

## 1. Original Objective

The project was initiated as a research-driven effort to build a miniature, fully-functional AQG system. The primary goals were:

- **Prototype a Multi-Agent Workflow**: Implement an end-to-end pipeline using orchestrated LLM agents to automatically generate educational questions from video transcripts.
- **Analyze and Solve Rejection Issues**: Specifically investigate why automated "Judge" agents might reject a high number of generated questions and build a "Fixer" loop to iteratively improve them.
- **Experiment with Modern Tooling**: Utilize `LangGraph` for workflow orchestration, `GPT-4o` for agent intelligence, and establish a framework for observability and cost-tracking.
- **Maintain Cost-Efficiency**: Keep the cost below a target of **$0.05 per generated question**.

The initial test bed was a set of 8 educational videos from the 3Blue1Brown YouTube channel.

## 2. What We Have Accomplished

We have successfully built and debugged the core AQG engine. The system is now fully functional.

### Key Achievements:

- **Complete End-to-End Workflow**: A `LangGraph`-based state machine successfully orchestrates the entire pipeline:
    1.  **Content Splitting**: The system ingests a long video transcript and intelligently splits it into smaller, coherent content segments.
    2.  **Summarization**: Each segment is then summarized to extract key concepts and generate specific, actionable learning objectives.
    3.  **Question Generation**: Using the summaries and learning objectives, the system generates relevant questions.
    4.  **Judging**: A sophisticated "Judge" agent evaluates each question against a 10-point rubric, providing a pass/fail decision and detailed feedback.
    5.  **Fixer Loop**: Rejected questions are automatically sent to a "Fixer" agent, which uses the judge's feedback to make improvements. The question is then sent back to the judge for re-evaluation. This loop runs for a maximum of 5 iterations.
    6.  **Results Finalization**: The workflow concludes by compiling all generated questions, evaluations, fix attempts, costs, and performance metrics into a single, detailed JSON file.

- **Successful Debugging & Refinement**:
    - Overcame initial challenges with LLM responses by implementing robust JSON cleaning and parsing logic.
    - Resolved Pydantic model validation errors by ensuring data structures match throughout the workflow.
    - Fixed critical bugs in the judge/fixer routing logic to ensure evaluations were correctly assigned and the fixer loop triggered appropriately.

- **Met Performance & Cost Targets**:
    - The system successfully processes a full video in minutes.
    - The total cost per run has been consistently around **$0.02**, which is **50-60% below** our target cost.

- **Robust Foundation**:
    - The project has a clean, modular structure.
    - All external API calls are centralized in a dedicated `LLMClient`.
    - A flexible prompt loading system allows for easy experimentation with different prompt versions.
    - Session-based logging provides a clear audit trail for each run.

## 3. Current Status

The core AQG engine is **complete and operational**. The system can reliably process a video from start to finish, producing high-quality questions while attempting to fix flawed ones.

We are now ready to build upon this solid foundation by adding user-facing features and improving the system's usability and scale. 