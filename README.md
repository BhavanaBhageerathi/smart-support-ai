# SmartSupport AI – Intelligent IT Support Ticket Triage System

## Overview

SmartSupport AI is an AI-powered IT support ticket triage system built using Django, LangGraph, Retrieval-Augmented Generation (RAG), and OpenAI.

The application analyzes IT support tickets, classifies issues, recommends priority levels and support teams, and generates troubleshooting recommendations using information retrieved from an IT knowledge base.

## Key Features

- **Automated Ticket Classification:** Classifies tickets into Network, Hardware, Software, Database, Security, Access Management, Email, and Other categories.
- **Priority Recommendation:** Recommends ticket priorities from P1 (Critical) to P4 (Low).
- **Team Recommendation:** Suggests the appropriate IT support team.
- **RAG-Based Knowledge Retrieval:** Retrieves relevant troubleshooting information from IT support documents using semantic search.
- **LangGraph Agent:** Uses multiple tools to coordinate ticket analysis and knowledge retrieval.
- **Structured AI Output:** Generates consistent triage results using Pydantic.
- **Django REST API:** Provides an API endpoint for submitting ticket details and receiving AI-generated results.

## Technology Stack

- Python
- Django
- Django REST Framework
- LangGraph
- LangChain
- OpenAI API
- ChromaDB
- Hugging Face Sentence Transformers
- Pydantic

## Project Structure

```text
SmartSupport_AI/
│
├── smartsupport/
├── triage/
├── knowledge_base/
├── manage.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## AI Workflow

1. Receive IT support ticket details.
2. Validate the incoming ticket data.
3. Use the LangGraph agent to coordinate the available tools.
4. Classify the ticket into an appropriate category.
5. Recommend a priority level and support team.
6. Retrieve relevant troubleshooting information from the knowledge base using ChromaDB.
7. Generate a structured triage result with recommended resolutions and human-intervention requirements.
8. Return the result through the Django REST API.

## RAG Implementation

The RAG pipeline uses IT troubleshooting documents stored in the `knowledge_base` directory.

- Documents are loaded from text files.
- Recursive Character Text Splitter divides documents into chunks.
- Hugging Face `all-MiniLM-L6-v2` generates embeddings.
- ChromaDB stores the document embeddings.
- Semantic similarity search retrieves the three most relevant chunks.
- Retrieved information is supplied to the LLM to generate troubleshooting recommendations.

## LangGraph Agent Tools

The agent uses four tools:

1. Knowledge Base Search
2. Ticket Classification
3. Priority Recommendation
4. Support Team Recommendation

LangGraph manages tool execution and coordinates the agent workflow.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-support-ai.git
cd smart-support-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

### 5. Run database migrations

```bash
python manage.py migrate
```

### 6. Start the Django development server

```bash
python manage.py runserver
```

## Configuration Notes

- The ChromaDB vector index is stored locally and can be generated from the documents in `knowledge_base/`.
- An OpenAI API key is required for LLM-based processing.
- The Hugging Face embedding model is downloaded when required.

## Future Improvements

- Ticket history and tracking.
- User authentication and role-based access.
- Ticket status management.
- Integration with external IT service management platforms.

## Author

Bhavana Bhageerathi
