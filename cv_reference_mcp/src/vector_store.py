"""
Sistema de Vetorização e Busca com FAISS

Este módulo implementa a vetorização de texto usando Sentence Transformers
e armazenamento/busca eficiente com FAISS para consultas semânticas.

Principais funcionalidades:
- Geração de embeddings com sentence-transformers
- Indexação e persistência com FAISS
- Busca por similaridade semântica
- Gerenciamento de metadados dos documentos
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from config import config
from pdf_processor import DocumentChunk


# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Resultado de uma busca semântica"""
    chunk_id: str
    text: str
    similarity_score: float
    page_number: int
    book_code: str
    book_title: str
    word_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte resultado para dicionário"""
        return self.dict()


class VectorStoreStats(BaseModel):
    """Estatísticas do vector store"""
    total_documents: int = 0
    index_dimension: int = 0
    model_name: str = ""
    index_type: str = ""
    last_updated: Optional[str] = None


class VectorStore:
    """Sistema de vetorização e busca semântica"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Inicializa o VectorStore
        
        Args:
            model_name: Nome do modelo sentence-transformers a usar
        """
        self.config = config
        self.model_name = model_name or config.embedding_model
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.stats = VectorStoreStats()
        
        # Inicializar modelo
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Inicializa o modelo sentence-transformers"""
        try:
            logger.info(f"Carregando modelo: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Modelo carregado: {self.model_name}")
            
            # Atualizar estatísticas
            self.stats.model_name = self.model_name
            self.stats.index_dimension = self.model.get_sentence_embedding_dimension()
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo {self.model_name}: {str(e)}")
            raise

    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Cria embeddings para uma lista de chunks
        
        Args:
            chunks: Lista de chunks de documentos
            
        Returns:
            Array numpy com os embeddings gerados
        """
        if not self.model:
            raise RuntimeError("Modelo não inicializado")
        
        if not chunks:
            return np.array([])
        
        logger.info(f"Gerando embeddings para {len(chunks)} chunks...")
        
        try:
            # Extrair textos dos chunks
            texts = [chunk.text for chunk in chunks]
            
            # Gerar embeddings
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,  # Processar em batches para economizar memória
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalizar para usar Inner Product
            )
            
            logger.info(f"✅ Embeddings gerados: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar embeddings: {str(e)}")
            raise

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Constrói índice FAISS com os chunks fornecidos
        
        Args:
            chunks: Lista de chunks para indexar
        """
        if not chunks:
            logger.warning("Nenhum chunk fornecido para indexação")
            return
        
        logger.info(f"Construindo índice FAISS para {len(chunks)} chunks...")
        
        try:
            # Gerar embeddings
            embeddings = self.create_embeddings(chunks)
            
            if embeddings.size == 0:
                logger.error("Nenhum embedding gerado")
                return
            
            # Criar índice FAISS
            dimension = embeddings.shape[1]
            
            # Usar IndexFlatIP (Inner Product) para embeddings normalizados
            # Isso é equivalente a cosine similarity para vetores normalizados
            self.index = faiss.IndexFlatIP(dimension)
            
            # Adicionar embeddings ao índice
            self.index.add(embeddings.astype(np.float32))
            
            # Preparar metadados
            self.metadata = [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "book_code": chunk.book_code,
                    "book_title": chunk.book_title,
                    "word_count": chunk.word_count,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                for chunk in chunks
            ]
            
            # Atualizar estatísticas
            self.stats.total_documents = len(chunks)
            self.stats.index_dimension = dimension
            self.stats.index_type = "IndexFlatIP"
            
            logger.info(f"✅ Índice FAISS construído com {len(chunks)} documentos")
            
        except Exception as e:
            logger.error(f"❌ Erro ao construir índice: {str(e)}")
            raise

    def save_index(self) -> None:
        """Salva o índice FAISS e metadados em disco"""
        if not self.index or not self.metadata:
            logger.warning("Nenhum índice ou metadados para salvar")
            return
        
        try:
            # Salvar índice FAISS
            faiss.write_index(self.index, str(config.faiss_index_path))
            logger.info(f"✅ Índice FAISS salvo: {config.faiss_index_path}")
            
            # Salvar metadados e estatísticas
            metadata_data = {
                "stats": self.stats.dict(),
                "metadata": self.metadata
            }
            
            with open(config.faiss_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Metadados salvos: {config.faiss_metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar índice: {str(e)}")
            raise

    def load_index(self) -> bool:
        """
        Carrega índice FAISS e metadados do disco
        
        Returns:
            True se carregado com sucesso, False caso contrário
        """
        try:
            # Verificar se arquivos existem
            if not config.faiss_index_path.exists():
                logger.info("Arquivo de índice FAISS não encontrado")
                return False
            
            if not config.faiss_metadata_path.exists():
                logger.info("Arquivo de metadados não encontrado")
                return False
            
            # Carregar índice FAISS
            self.index = faiss.read_index(str(config.faiss_index_path))
            logger.info(f"✅ Índice FAISS carregado: {config.faiss_index_path}")
            
            # Carregar metadados
            with open(config.faiss_metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.stats = VectorStoreStats(**data.get("stats", {}))
            self.metadata = data.get("metadata", [])
            
            logger.info(f"✅ Metadados carregados: {len(self.metadata)} documentos")
            
            # Verificar consistência
            if self.index.ntotal != len(self.metadata):
                logger.warning(
                    f"Inconsistência: {self.index.ntotal} vetores vs "
                    f"{len(self.metadata)} metadados"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar índice: {str(e)}")
            return False

    def search(
        self, 
        query: str, 
        top_k: int = None, 
        similarity_threshold: float = None,
        book_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Realiza busca semântica
        
        Args:
            query: Texto da consulta
            top_k: Número máximo de resultados (default: config.default_top_k)
            similarity_threshold: Threshold mínimo de similaridade
            book_filter: Filtrar por código de livro específico
            
        Returns:
            Lista de resultados ordenados por relevância
        """
        if not self.model or not self.index or not self.metadata:
            logger.error("VectorStore não inicializado ou vazio")
            return []
        
        # Usar valores padrão se não especificados
        top_k = top_k or config.default_top_k
        similarity_threshold = similarity_threshold or config.similarity_threshold
        
        # Limitar top_k ao máximo configurado
        top_k = min(top_k, config.max_top_k)
        
        try:
            # Gerar embedding da query
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Buscar no índice FAISS
            # Para IndexFlatIP com embeddings normalizados, scores são cosine similarity
            scores, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(top_k * 2, len(self.metadata))  # Buscar mais para filtrar depois
            )
            
            # Processar resultados
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Verificar se o índice é válido
                if idx == -1 or idx >= len(self.metadata):
                    continue
                
                # Aplicar threshold de similaridade
                if score < similarity_threshold:
                    continue
                
                # Obter metadados
                metadata = self.metadata[idx]
                
                # Aplicar filtro de livro se especificado
                if book_filter and metadata["book_code"] != book_filter:
                    continue
                
                # Criar resultado
                result = SearchResult(
                    chunk_id=metadata["chunk_id"],
                    text=metadata["text"],
                    similarity_score=float(score),
                    page_number=metadata["page_number"],
                    book_code=metadata["book_code"],
                    book_title=metadata["book_title"],
                    word_count=metadata["word_count"]
                )
                
                results.append(result)
                
                # Parar quando atingir top_k resultados válidos
                if len(results) >= top_k:
                    break
            
            logger.info(f"Busca retornou {len(results)} resultados para: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro na busca: {str(e)}")
            return []

    def get_document_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Busca documento específico por ID
        
        Args:
            chunk_id: ID do chunk
            
        Returns:
            Metadados do documento ou None se não encontrado
        """
        for metadata in self.metadata:
            if metadata["chunk_id"] == chunk_id:
                return metadata
        return None

    def list_books(self) -> Dict[str, Dict[str, Any]]:
        """
        Lista informações sobre os livros indexados
        
        Returns:
            Dicionário com estatísticas por livro
        """
        books_stats = {}
        
        for metadata in self.metadata:
            book_code = metadata["book_code"]
            
            if book_code not in books_stats:
                books_stats[book_code] = {
                    "book_title": metadata["book_title"],
                    "chunk_count": 0,
                    "total_words": 0,
                    "pages": set()
                }
            
            books_stats[book_code]["chunk_count"] += 1
            books_stats[book_code]["total_words"] += metadata["word_count"]
            books_stats[book_code]["pages"].add(metadata["page_number"])
        
        # Converter sets para listas para serialização JSON
        for book_code in books_stats:
            books_stats[book_code]["page_count"] = len(books_stats[book_code]["pages"])
            books_stats[book_code]["pages"] = sorted(list(books_stats[book_code]["pages"]))
        
        return books_stats

    def get_stats(self) -> VectorStoreStats:
        """Retorna estatísticas do vector store"""
        return self.stats

    def is_ready(self) -> bool:
        """Verifica se o vector store está pronto para uso"""
        return (
            self.model is not None and 
            self.index is not None and 
            len(self.metadata) > 0
        )