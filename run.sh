#!/bin/bash

export CLAUDE_CODE_USE_FOUNDRY="1" 
export ANTHROPIC_FOUNDRY_BASE_URL="" 
export ANTHROPIC_FOUNDRY_API_KEY="" 
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-8"

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your ANTHROPIC_API_KEY in .env"

# Change to backend directory and start the server
cd backend && uv run uvicorn app:app --reload --port 8000