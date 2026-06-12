import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG system"""

    # Azure Anthropic Foundry settings
    AZURE_API_KEY: str = os.getenv("ANTHROPIC_FOUNDRY_API_KEY", "")
    AZURE_ENDPOINT: str = os.getenv("ANTHROPIC_FOUNDRY_BASE_URL", "")
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800  # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100  # Characters to overlap between chunks
    MAX_RESULTS: int = 5  # Maximum search results to return
    MAX_HISTORY: int = 2  # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location


config = Config()
