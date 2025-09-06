"""
Handler de Consultas Semânticas

Este módulo processa e otimiza consultas do usuário, fornecendo uma interface
inteligente para busca nas referências bibliográficas de Visão Computacional.

Principais funcionalidades:
- Processamento e normalização de queries
- Expansão de consultas com sinônimos de CV
- Formatação de resultados para o MCP
- Contexto inteligente para respostas
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from .config import config
from .vector_store import VectorStore, SearchResult


# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Tipos de consulta identificados"""
    CONCEPT = "concept"  # Sobre conceito específico
    ALGORITHM = "algorithm"  # Sobre algoritmo específico
    IMPLEMENTATION = "implementation"  # Como implementar algo
    THEORY = "theory"  # Teoria/matemática
    COMPARISON = "comparison"  # Comparação entre métodos
    GENERAL = "general"  # Consulta geral


@dataclass
class QueryContext:
    """Contexto da consulta processada"""
    original_query: str
    processed_query: str
    query_type: QueryType
    detected_concepts: List[str]
    suggested_book: Optional[str]
    confidence: float


class FormattedResponse(BaseModel):
    """Resposta formatada para o MCP"""
    query: str
    total_results: int
    results: List[Dict[str, Any]]
    context: Dict[str, Any]
    suggestions: List[str]


class QueryHandler:
    """Processador inteligente de consultas"""
    
    def __init__(self, vector_store: VectorStore):
        """
        Inicializa o handler de consultas
        
        Args:
            vector_store: Instância do VectorStore para busca
        """
        self.vector_store = vector_store
        self.config = config
        
        # Dicionário de sinônimos e termos relacionados para CV
        self.cv_synonyms = {
            # Processamento básico
            "imagem": ["image", "figura", "picture", "foto"],
            "pixel": ["elemento", "ponto", "sample"],
            "resolução": ["resolution", "tamanho", "dimensão"],
            
            # Espaços de cor
            "rgb": ["red green blue", "vermelho verde azul"],
            "hsv": ["hue saturation value", "matiz saturação valor"],
            "cinza": ["gray", "grayscale", "escala de cinza"],
            
            # Filtros
            "gaussiano": ["gaussian", "gauss", "suavização"],
            "sobel": ["detecção borda", "edge detection", "gradiente"],
            "laplaciano": ["laplacian", "segunda derivada"],
            "média": ["average", "mean", "suavização"],
            
            # Morfologia
            "erosão": ["erosion", "erode", "morfologia"],
            "dilatação": ["dilation", "dilate", "morfologia"],
            "abertura": ["opening", "morfologia"],
            "fechamento": ["closing", "morfologia"],
            
            # Segmentação
            "limiarização": ["thresholding", "threshold", "binarização"],
            "otsu": ["limiar automático", "automatic threshold"],
            "canny": ["detecção borda", "edge detection"],
            "watershed": ["segmentação", "bacias hidrográficas"],
            
            # Características
            "sift": ["scale invariant", "pontos interesse", "keypoints"],
            "surf": ["speeded up", "pontos interesse"],
            "hog": ["histogram oriented gradients", "gradientes"],
            "lbp": ["local binary patterns", "padrões binários"],
            
            # Deep Learning
            "cnn": ["convolutional neural", "rede neural convolucional"],
            "pooling": ["agrupamento", "redução dimensional"],
            "dropout": ["regularização", "regularization"],
            "backpropagation": ["retropropagação", "treinamento"],
            
            # Objetos e detecção
            "yolo": ["you only look once", "detecção objetos"],
            "rcnn": ["region cnn", "detecção objetos"],
            "mask rcnn": ["segmentação instância", "instance segmentation"],
        }
        
        # Padrões para identificar tipo de consulta
        self.query_patterns = {
            QueryType.CONCEPT: [
                r"o que é", r"what is", r"define", r"definição",
                r"conceito", r"significa", r"explique"
            ],
            QueryType.ALGORITHM: [
                r"algoritmo", r"algorithm", r"método", r"técnica",
                r"como funciona", r"funcionamento"
            ],
            QueryType.IMPLEMENTATION: [
                r"como implementar", r"implementação", r"código",
                r"como fazer", r"passos", r"tutorial"
            ],
            QueryType.THEORY: [
                r"matemática", r"fórmula", r"equação", r"teoria",
                r"mathematical", r"formula", r"equation"
            ],
            QueryType.COMPARISON: [
                r"diferença", r"vs", r"versus", r"comparar",
                r"melhor que", r"difference", r"compare"
            ],
        }
        
        # Mapeamento de conceitos para livros mais relevantes
        self.concept_to_book = {
            # Fundamentos (Gonzalez)
            "convolução": "gonzalez",
            "filtro": "gonzalez",
            "histograma": "gonzalez",
            "limiarização": "gonzalez",
            "morfologia": "gonzalez",
            
            # Visão computacional moderna (Szeliski)
            "câmera": "szeliski",
            "calibração": "szeliski",
            "stereo": "szeliski",
            "movimento": "szeliski",
            
            # Deep Learning (Goodfellow)
            "backpropagation": "goodfellow",
            "otimização": "goodfellow",
            "regularização": "goodfellow",
            "dropout": "goodfellow",
            
            # Machine Learning (Bishop)
            "bayes": "bishop",
            "probabilidade": "bishop",
            "classificação": "bishop",
            "clustering": "bishop",
        }

    def process_query(self, query: str) -> QueryContext:
        """
        Processa e analisa a consulta do usuário
        
        Args:
            query: Consulta original do usuário
            
        Returns:
            Contexto da consulta processada
        """
        # Normalizar query
        processed_query = self._normalize_query(query)
        
        # Detectar tipo da consulta
        query_type = self._detect_query_type(processed_query)
        
        # Detectar conceitos de CV
        detected_concepts = self._detect_cv_concepts(processed_query)
        
        # Sugerir livro mais relevante
        suggested_book = self._suggest_book(detected_concepts, processed_query)
        
        # Calcular confiança da análise
        confidence = self._calculate_confidence(query_type, detected_concepts)
        
        context = QueryContext(
            original_query=query,
            processed_query=processed_query,
            query_type=query_type,
            detected_concepts=detected_concepts,
            suggested_book=suggested_book,
            confidence=confidence
        )
        
        logger.info(f"Query processada: {query_type.value}, conceitos: {detected_concepts}")
        return context

    def search(
        self,
        query: str,
        top_k: int = None,
        book_filter: Optional[str] = None,
        similarity_threshold: float = None
    ) -> FormattedResponse:
        """
        Realiza busca semântica e formata resposta
        
        Args:
            query: Consulta do usuário
            top_k: Número máximo de resultados
            book_filter: Filtro por livro específico
            similarity_threshold: Threshold de similaridade
            
        Returns:
            Resposta formatada com resultados e contexto
        """
        # Processar consulta
        context = self.process_query(query)
        
        # Expandir consulta com sinônimos
        expanded_query = self._expand_query(context.processed_query)
        
        # Usar parâmetros padrão se não especificados
        top_k = top_k or config.default_top_k
        similarity_threshold = similarity_threshold or config.similarity_threshold
        
        # Aplicar filtro de livro sugerido se não especificado
        if not book_filter and context.suggested_book:
            book_filter = context.suggested_book
            # Reduzir threshold se usando livro sugerido
            similarity_threshold *= 0.9
        
        # Realizar busca
        search_results = self.vector_store.search(
            query=expanded_query,
            top_k=top_k,
            book_filter=book_filter,
            similarity_threshold=similarity_threshold
        )
        
        # Se não encontrou resultados com filtro, tentar sem filtro
        if not search_results and book_filter:
            logger.info("Tentando busca sem filtro de livro...")
            search_results = self.vector_store.search(
                query=expanded_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        
        # Formatar resposta
        response = self._format_response(
            query=context.original_query,
            results=search_results,
            context=context
        )
        
        return response

    def _normalize_query(self, query: str) -> str:
        """Normaliza a consulta removendo caracteres especiais e padronizando"""
        # Converter para minúsculas
        query = query.lower().strip()
        
        # Remover pontuação desnecessária
        query = re.sub(r'[^\w\s\-]', ' ', query)
        
        # Normalizar espaços
        query = re.sub(r'\s+', ' ', query)
        
        return query

    def _detect_query_type(self, query: str) -> QueryType:
        """Detecta o tipo da consulta baseado em padrões"""
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return query_type
        
        return QueryType.GENERAL

    def _detect_cv_concepts(self, query: str) -> List[str]:
        """Detecta conceitos de CV mencionados na consulta"""
        detected = []
        
        # Verificar conceitos configurados
        for concept in config.cv_concepts:
            concept_clean = concept.replace('-', ' ')
            if concept_clean in query or concept in query:
                detected.append(concept)
        
        # Verificar sinônimos
        for main_term, synonyms in self.cv_synonyms.items():
            if main_term in query:
                detected.append(main_term)
            else:
                for synonym in synonyms:
                    if synonym.lower() in query:
                        detected.append(main_term)
                        break
        
        return list(set(detected))  # Remover duplicatas

    def _suggest_book(self, concepts: List[str], query: str) -> Optional[str]:
        """Sugere livro mais relevante baseado nos conceitos detectados"""
        book_scores = {}
        
        # Pontuar baseado em conceitos
        for concept in concepts:
            suggested = self.concept_to_book.get(concept)
            if suggested:
                book_scores[suggested] = book_scores.get(suggested, 0) + 1
        
        # Pontuar baseado em palavras-chave na query
        book_keywords = {
            "gonzalez": ["processamento", "filtro", "transformada", "morfologia"],
            "szeliski": ["visão", "câmera", "estéreo", "movimento", "3d"],
            "goodfellow": ["deep", "neural", "learning", "cnn", "rede"],
            "bishop": ["machine", "pattern", "probabilidade", "bayes", "classificação"]
        }
        
        for book, keywords in book_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    book_scores[book] = book_scores.get(book, 0) + 0.5
        
        # Retornar livro com maior pontuação
        if book_scores:
            return max(book_scores.items(), key=lambda x: x[1])[0]
        
        return None

    def _calculate_confidence(self, query_type: QueryType, concepts: List[str]) -> float:
        """Calcula confiança da análise da consulta"""
        confidence = 0.5  # Base
        
        # Bonus por tipo específico identificado
        if query_type != QueryType.GENERAL:
            confidence += 0.2
        
        # Bonus por conceitos detectados
        confidence += min(len(concepts) * 0.1, 0.3)
        
        return min(confidence, 1.0)

    def _expand_query(self, query: str) -> str:
        """Expande a consulta com sinônimos e termos relacionados"""
        expanded_terms = [query]
        
        # Adicionar sinônimos de termos encontrados
        for main_term, synonyms in self.cv_synonyms.items():
            if main_term in query:
                # Adicionar alguns sinônimos principais
                expanded_terms.extend(synonyms[:2])
        
        # Juntar termos expandidos
        expanded_query = " ".join(expanded_terms)
        
        # Limitar tamanho da query expandida
        if len(expanded_query) > 500:
            expanded_query = expanded_query[:500]
        
        return expanded_query

    def _format_response(
        self,
        query: str,
        results: List[SearchResult],
        context: QueryContext
    ) -> FormattedResponse:
        """Formata a resposta final para o MCP"""
        
        # Preparar resultados formatados
        formatted_results = []
        for result in results:
            formatted_result = {
                "chunk_id": result.chunk_id,
                "text": self._truncate_text(result.text, 800),  # Limitar texto
                "similarity_score": round(result.similarity_score, 3),
                "page_number": result.page_number,
                "book_code": result.book_code,
                "book_title": result.book_title,
                "word_count": result.word_count,
                "source": f"{result.book_title}, página {result.page_number}"
            }
            formatted_results.append(formatted_result)
        
        # Preparar contexto da resposta
        response_context = {
            "query_type": context.query_type.value,
            "detected_concepts": context.detected_concepts,
            "suggested_book": context.suggested_book,
            "confidence": round(context.confidence, 2),
            "total_books_searched": len(self.vector_store.list_books()),
        }
        
        # Gerar sugestões
        suggestions = self._generate_suggestions(context, results)
        
        return FormattedResponse(
            query=query,
            total_results=len(results),
            results=formatted_results,
            context=response_context,
            suggestions=suggestions
        )

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Trunca texto preservando palavras completas"""
        if len(text) <= max_length:
            return text
        
        # Encontrar última palavra completa antes do limite
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Se não perdeu muito, usar
            return truncated[:last_space] + "..."
        
        return truncated + "..."

    def _generate_suggestions(
        self,
        context: QueryContext,
        results: List[SearchResult]
    ) -> List[str]:
        """Gera sugestões de consultas relacionadas"""
        suggestions = []
        
        # Sugerir conceitos relacionados não explorados
        all_concepts = set(config.cv_concepts)
        used_concepts = set(context.detected_concepts)
        unused_concepts = all_concepts - used_concepts
        
        # Selecionar conceitos relacionados baseado nos resultados
        if results:
            # Analisar texto dos resultados para encontrar conceitos mencionados
            result_text = " ".join([r.text[:200] for r in results[:3]])
            related_concepts = []
            
            for concept in unused_concepts:
                concept_clean = concept.replace('-', ' ')
                if concept_clean in result_text.lower():
                    related_concepts.append(concept_clean)
            
            # Adicionar algumas sugestões baseadas em conceitos relacionados
            for concept in related_concepts[:3]:
                suggestions.append(f"Como funciona {concept}?")
                suggestions.append(f"Implementação de {concept}")
        
        # Sugestões baseadas no tipo de query
        if context.query_type == QueryType.CONCEPT and context.detected_concepts:
            concept = context.detected_concepts[0]
            suggestions.extend([
                f"Como implementar {concept}",
                f"Exemplo prático de {concept}",
                f"Parâmetros para {concept}"
            ])
        
        elif context.query_type == QueryType.ALGORITHM:
            suggestions.extend([
                "Vantagens e desvantagens do algoritmo",
                "Comparação com outros métodos",
                "Implementação passo a passo"
            ])
        
        # Limitar número de sugestões
        return suggestions[:5]

    def get_concept_info(self, concept: str) -> Optional[FormattedResponse]:
        """
        Busca informações específicas sobre um conceito
        
        Args:
            concept: Nome do conceito a pesquisar
            
        Returns:
            Resposta formatada ou None se conceito não encontrado
        """
        # Normalizar conceito
        concept_normalized = concept.lower().replace('_', ' ').replace('-', ' ')
        
        # Verificar se é conceito conhecido
        if concept_normalized not in [c.replace('-', ' ') for c in config.cv_concepts]:
            return None
        
        # Buscar informações sobre o conceito
        query = f"definição conceito {concept_normalized} explicação teoria"
        return self.search(query, top_k=3)

    def list_available_concepts(self) -> List[str]:
        """Lista conceitos disponíveis para consulta"""
        return sorted(config.cv_concepts)

    def get_book_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas dos livros indexados"""
        return self.vector_store.list_books()