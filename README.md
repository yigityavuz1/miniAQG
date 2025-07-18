# miniAQG - Miniature Automated Question Generation System

A lightweight, cost-effective automated question generation system that creates educational questions from video transcripts using orchestrated LLM agents.

## 🎯 Overview

miniAQG is a prototype system that demonstrates a multi-agent workflow for generating high-quality educational questions from video content. The system uses a judge-fixer loop to ensure question quality and maintains cost efficiency under $0.05 per question.

## ✨ Features

- **Multi-Agent Workflow**: Content splitting, summarization, question generation, judging, and fixing
- **Quality Assurance**: Judge agent evaluates questions and fixer agent improves rejected ones
- **Cost Efficient**: Optimized to stay under $0.05 per question
- **REST API**: Full API endpoints for external integration
- **Modular Design**: Clean separation of concerns with centralized configuration
- **Session Logging**: Comprehensive logging for debugging and analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Content       │    │   Question      │    │   Judge Agent   │
│   Splitter      │───▶│   Generator     │───▶│   (Quality      │
│                 │    │                 │    │   Check)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Results       │◀───│   Fixer Agent   │◀───│   Rejected      │
│   Finalizer     │    │   (Improve      │    │   Questions     │
│                 │    │   Questions)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for GPT-4o)
- Google GenAI API key (for Gemini)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yigityavuz1/miniAQG.git
   cd miniAQG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the API server**
   ```bash
   python src/api/main.py
   ```

## 📚 API Endpoints

### Core Endpoints

- `POST /generate` - Generate questions from content
- `POST /evaluate` - Evaluate question quality
- `POST /fix` - Fix rejected questions
- `POST /workflow` - Run complete workflow
- `GET /status/{workflow_id}` - Check workflow status
- `GET /results/{workflow_id}` - Retrieve workflow results

### Example Usage

```python
import requests

# Generate questions
response = requests.post("http://localhost:8000/generate", json={
    "content": "Your video transcript here...",
    "question_type": "multiple_choice",
    "difficulty": "medium",
    "num_questions": 3
})

# Evaluate a question
response = requests.post("http://localhost:8000/evaluate", json={
    "question": {
        "text": "What is the main topic?",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "A"
    },
    "context": "Relevant context..."
})
```

## 🔧 Configuration

The system uses centralized configuration through `config.yaml`:

```yaml
api_keys:
  openai: "your-openai-key"
  google_genai: "your-google-key"

models:
  main: "gpt-4o"
  research: "gemini-1.5-pro"

cost_limits:
  max_per_question: 0.05
  max_per_workflow: 1.00
```

## 📊 Performance

- **Cost**: ~$0.02 per question generation
- **Quality**: Judge-fixer loop ensures high-quality output
- **Speed**: Optimized for quick response times
- **Reliability**: Robust error handling and retry mechanisms

## 🧪 Testing

Run the test scripts to verify functionality:

```bash
# Test question generation
python test_generate.py

# Test evaluation
python test_evaluate.py

# Test the complete workflow
python test_workflow.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Powered by OpenAI GPT-4o and Google Gemini
- Inspired by educational content creators like 3Blue1Brown

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a prototype system designed for educational and research purposes. For production use, additional security, scalability, and reliability measures should be implemented. 