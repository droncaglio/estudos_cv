"""
CV Reference MCP Server

Um servidor MCP (Model Context Protocol) para consulta semântica de referências
bibliográficas de Visão Computacional usando RAG (Retrieval-Augmented Generation).

Este módulo permite:
- Extração e processamento de PDFs de referências
- Vetorização de conteúdo usando sentence-transformers
- Busca semântica com FAISS
- Integração com Claude Code via MCP
"""

__version__ = "0.1.0"
__author__ = "Estudos CV"
__email__ = "estudos@cv.local"

from .server import app as mcp_server
from .config import Config
from .pdf_processor import PDFProcessor
from .vector_store import VectorStore
from .query_handler import QueryHandler

__all__ = [
    "mcp_server",
    "Config", 
    "PDFProcessor",
    "VectorStore",
    "QueryHandler",
]