# Insurance Assistant Agent

An intelligent RAG (Retrieval-Augmented Generation) based chat agent that helps customers with car insurance claims by leveraging web-crawled content, vector databases, and Azure OpenAI.

[Chat with the agent here](https://huggingface.co/spaces/shobanapriyan/src)
[Insurance company details](https://www.allianz.co.uk/insurance/car-insurance/existing-customers/claim.html)

## Overview

This project implements an AI-powered insurance assistant that can answer questions about car insurance claims by:
1. Crawling and extracting content from insurance company websites
2. Processing and storing the content in a vector database
3. Retrieving relevant context for user queries
4. Generating accurate, context-aware responses using Azure OpenAI

## Features

- **Automated Web Crawling**: Extracts insurance claim information from Allianz UK website
- **Intelligent Context Retrieval**: Uses FAISS vector database for semantic search
- **Natural Conversations**: Azure OpenAI-powered chat interface
- **User-Friendly UI**: Built with Gradio for easy interaction
- **RAG Architecture**: Combines retrieval and generation for accurate responses

## Architecture

```
┌─────────────────┐
│  Web Crawler    │ ──> Extracts insurance content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Processing │ ──> Chunks content for embedding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Database │ ──> FAISS index for semantic search
│    (FAISS)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chat Interface │ ──> Retrieves context + generates response
│  (Gradio + GPT) │
└─────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/s-priyan/Insurance-Assistant-Agent.git
cd insurance-assistant-agent
```

2. **Set up environment variables**

Create a `.env` file in the root directory with your credentials


## Usage

### Step 1: Crawl Insurance Website

Extract content from the Allianz car insurance claims page:

```bash
cd src
uv run web_crawler.py
```

This will save the crawled content to `data/content.md`.

### Step 2: Build Knowledge Base

Process the content and create a vector database:

```bash
uv run long_term_memory.py
```

This creates a FAISS index in the `vectore/insurance` directory.

### Step 3: Launch Chat Interface

Start the Gradio UI:

```bash
uv run workflow_ui.py
```

The interface will be available at `http://localhost:7860`

### Using the Chatbot

Once the interface is running:
1. Ask questions about car insurance claims
2. Get instant, contextually accurate responses
3. Continue natural conversations with history tracking

**Example Questions:**
- "How do I file a car insurance claim?"
- "What documents do I need for a claim?"
- "How long does claim processing take?"

## Project Structure

```
insurance-assistant-agent/
│
├── src/
│   ├── web_crawler.py           # Web scraping module
│   ├── long_term_memory.py      # Vector database creation
│   └── workflow_ui.py            # Chat interface
│
├── tests/
│   ├── test_web_crawler.py      # Unit tests for crawler
│   ├── test_long_term_memory.py # Unit tests for memory
│   ├── test_workflow_ui.py      # Unit tests for UI
│
├── data/
│   └── content.md               # Crawled content (generated)
│
├── vectore/
│   └── insurance/               # FAISS index (generated)
│
├── .env                         # Environment variables (create this)
├── .gitignore                   # Git ignore file
├── pyproject.toml               # Project dependencies
└── README.md                    # Project description
```

## Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Gradio Documentation](https://www.gradio.app/docs/)