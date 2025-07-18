# Development Plan: API, UI, and Batch Processing

This document outlines the plan for the next major development phases, building upon the completed core AQG engine.

---

## 1. 🌐 Phase 1: API Endpoints

This phase focuses on exposing the core AQG agent functionalities through a `FastAPI` server, enabling external interaction and integration.

### 1.1. Core Endpoints (Must-Have)

-   **`POST /questions/generate`**
    -   **Purpose**: Generate a single question on-demand.
    -   **Request Body**: `{ "learning_objective": "...", "content_summary": "..." }`
    -   **Action**: Directly invokes the `QuestionGenerator` agent logic.
    -   **Response**: A `Question` object.

-   **`POST /questions/evaluate`**
    -   **Purpose**: Evaluate a single question on-demand.
    -   **Request Body**: A `Question` object.
    -   **Action**: Directly invokes the `Judge` agent logic.
    -   **Response**: A `JudgeEvaluation` object.

-   **`POST /questions/fix`**
    -   **Purpose**: Attempt to fix a single rejected question.
    -   **Request Body**: `{ "rejected_question": Question, "evaluation": JudgeEvaluation }`
    -   **Action**: Directly invokes the `Fixer` agent logic.
    -   **Response**: A `Question` object representing the fixed version.

### 1.2. Workflow Endpoints (Recommended)

-   **`POST /workflow/run/{video_id}`**
    -   **Purpose**: Trigger the full, end-to-end AQG workflow for a specified video.
    -   **Action**: Initializes and runs the `AQGWorkflow.process_video` method asynchronously.
    -   **Response**: `{ "session_id": "...", "status": "started" }`

-   **`GET /workflow/status/{session_id}`**
    -   **Purpose**: Check the status of a running workflow.
    -   **Action**: Queries the state of the LangGraph workflow instance.
    -   **Response**: `{ "session_id": "...", "current_stage": "...", "total_cost": "..." }`

-   **`GET /results/{video_id}`**
    -   **Purpose**: Retrieve the final results of a completed workflow.
    -   **Action**: Reads the corresponding results JSON file from the `outputs/` directory.
    -   **Response**: The `VideoProcessingResult` object.

### 1.3. Implementation Details

-   **Technology**: `FastAPI`
-   **File Structure**: Code will be placed in `src/api/`.
-   **Models**: `Pydantic` models will be used for all request/response bodies to ensure type safety and clear documentation. These can reuse or extend the models in `src/workflows/state_models.py`.

---

## 2. 📊 Phase 2: Dashboard and UI

This phase focuses on creating a simple, real-time monitoring dashboard to provide visibility into the running system.

### 2.1. Core Features

-   **Real-Time Workflow Monitoring**:
    -   **Purpose**: Visualize the progress of AQG workflows as they execute.
    -   **Implementation**: A `Streamlit` application that connects to a `WebSocket` server. The `AQGWorkflow` nodes will be modified to emit events (e.g., `node_start`, `node_end`, `error`) to the WebSocket.
    -   **UI**: Display the graph of the workflow, highlighting the currently active node. Show logs, costs, and token counts in real-time.

-   **Results Viewer**:
    -   **Purpose**: Provide a user-friendly way to inspect the final results of a video.
    -   **Implementation**: A section in the Streamlit app that allows selecting a video ID and then neatly displays the contents of the `_results.json` file.
    -   **UI**: Show approved questions, rejected questions (with judge feedback), and key metrics in tables and charts.

-   **Interactive Agent Playground**:
    -   **Purpose**: Allow for manual testing and debugging of individual agents.
    -   **Implementation**: Use Streamlit input widgets (text areas, buttons) to call the API endpoints from Phase 1.
    -   **UI**: Provide a form to generate a question, another to evaluate one, and a third to submit a question for fixing. This is invaluable for rapid prompt engineering.

### 2.2. Technology Stack

-   **UI Framework**: `Streamlit`
-   **Real-Time Communication**: `FastAPI WebSockets`
-   **File Structure**: Code will be placed in `src/dashboard/` and `src/api/websocket.py`.

---

## 3. 🔄 Phase 3: Batch Processing

This phase enhances the system to process multiple videos efficiently.

### 3.1. Core Features

-   **Concurrent Video Processing**:
    -   **Purpose**: Run the AQG workflow for all 8 videos in parallel to save time.
    -   **Implementation**: Modify `main.py` to accept a `--batch` or `--all` flag. Use `asyncio.gather` to execute multiple `AQGWorkflow.process_video` coroutines concurrently.
    -   **Output**: Individual result files will be generated for each video.

-   **Aggregate Metrics**:
    -   **Purpose**: After a batch run, calculate and display metrics across all videos.
    -   **Implementation**: A new script or function that reads all result files from the batch run and computes aggregate statistics (total cost, overall approval rate, etc.).

---

## 4. 📈 Phase 4: Analytics and Reporting

This phase focuses on deriving deeper insights from the data generated by the system.

### 4.1. Core Features

-   **Automated Reporting**:
    -   **Purpose**: Generate a summary report after each batch run.
    -   **Implementation**: A new module (`src/core/analytics.py`) that loads all result files from a session.
    -   **Output**: A markdown file (`reports/summary_{session_id}.md`) containing:
        -   Overall cost and performance metrics.
        -   A breakdown of question approval/rejection rates.
        -   Analysis of the most common reasons for rejection (by parsing `JudgeEvaluation` feedback).
        -   A list of the "most difficult" questions (those requiring the most fix attempts).

-   **Dashboard Integration**:
    -   **Purpose**: Display the analytics report in the UI.
    -   **Implementation**: The Streamlit app will have an "Analytics" tab where users can select and view these generated reports. 