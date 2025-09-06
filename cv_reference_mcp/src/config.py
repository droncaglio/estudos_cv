"""
Configurações do CV Reference MCP Server

Define configurações centralizadas para o servidor MCP, incluindo paths,
modelos, parâmetros de busca e outras configurações essenciais.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configurações principais do MCP server"""
    
    # Paths do projeto
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    pdfs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "pdfs")
    processed_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    vectors_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "vectors")
    
    # Configurações de embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # Dimensão do modelo all-MiniLM-L6-v2
    
    # Configurações de processamento de PDF
    chunk_size: int = 512  # Tamanho dos chunks em caracteres
    chunk_overlap: int = 50  # Sobreposição entre chunks
    min_chunk_size: int = 100  # Tamanho mínimo do chunk
    
    # Configurações de busca
    default_top_k: int = 5  # Número padrão de resultados
    max_top_k: int = 20  # Máximo de resultados por consulta
    similarity_threshold: float = 0.3  # Threshold mínimo de similaridade
    
    # Configurações FAISS
    faiss_index_type: str = "IndexFlatIP"  # Inner Product para sentence-transformers
    faiss_index_file: str = "cv_references.index"
    faiss_metadata_file: str = "cv_references_metadata.json"
    
    # Mapeamento de livros de referência
    reference_books: Dict[str, str] = {
        "gonzalez": "Digital Image Processing - Gonzalez & Woods",
        "szeliski": "Computer Vision Algorithms and Applications - Szeliski", 
        "goodfellow": "Deep Learning - Goodfellow, Bengio & Courville",
        "bishop": "Pattern Recognition and Machine Learning - Bishop",
    }
    
    # Conceitos-chave de CV (para queries específicas)
    cv_concepts: List[str] = [
        # Fundamentos
        "amostragem", "quantização", "imagem-digital", "pixel",
        "rgb", "hsv", "escala-cinza", "espaços-cor",
        
        # Processamento de imagens
        "convolução", "filtro-gaussiano", "filtro-media", 
        "sobel", "prewitt", "laplaciano", "detecção-bordas",
        "histograma", "equalização", "clahe",
        
        # Morfologia
        "erosão", "dilatação", "abertura", "fechamento",
        "morfologia-matemática", "watershed",
        
        # Segmentação
        "limiarização", "otsu", "canny", "hough",
        "segmentação", "crescimento-regiões",
        
        # Características
        "sift", "surf", "orb", "fast", "hog", "lbp",
        "descritores", "pontos-interesse", "matching",
        
        # ML e Deep Learning
        "cnn", "redes-neurais", "backpropagation",
        "dropout", "batch-normalization", "transfer-learning",
        "yolo", "rcnn", "unet", "segmentação-semântica",
    ]
    
    # Configurações do servidor MCP
    server_name: str = "cv-reference-mcp"
    server_version: str = "0.1.0"
    server_description: str = "Servidor MCP para consulta de referências de Visão Computacional"
    
    def model_post_init(self, __context: Any) -> None:
        """Criar diretórios necessários após inicialização"""
        for directory in [self.data_dir, self.pdfs_dir, self.processed_dir, self.vectors_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def faiss_index_path(self) -> Path:
        """Path completo do índice FAISS"""
        return self.vectors_dir / self.faiss_index_file
    
    @property
    def faiss_metadata_path(self) -> Path:
        """Path completo dos metadados FAISS"""
        return self.vectors_dir / self.faiss_metadata_file
    
    def get_book_code(self, book_name: str) -> str:
        """Obter código do livro a partir do nome completo"""
        for code, full_name in self.reference_books.items():
            if book_name.lower() in full_name.lower():
                return code
        return "unknown"
    
    def get_book_name(self, book_code: str) -> str:
        """Obter nome completo do livro a partir do código"""
        return self.reference_books.get(book_code.lower(), f"Unknown Book ({book_code})")


# Instância global da configuração
config = Config()