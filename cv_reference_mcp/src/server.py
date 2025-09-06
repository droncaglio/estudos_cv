"""
Servidor MCP Principal para Consulta de Referências de CV

Este módulo implementa o servidor MCP (Model Context Protocol) que expõe
recursos e ferramentas para consulta semântica das referências bibliográficas
de Visão Computacional.

Funcionalidades disponíveis:
- Recursos para busca semântica e conceitos específicos
- Ferramentas para consultas interativas
- Integração com Claude Code via MCP
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from .config import config
from .pdf_processor import PDFProcessor
from .vector_store import VectorStore
from .query_handler import QueryHandler, FormattedResponse


# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Modelos para validação de entrada
class QueryRequest(BaseModel):
    """Modelo para requisições de consulta"""
    query: str
    top_k: Optional[int] = None
    book_filter: Optional[str] = None
    similarity_threshold: Optional[float] = None


class ConceptRequest(BaseModel):
    """Modelo para requisições de conceito específico"""
    concept: str


# Instância global do servidor MCP
mcp = FastMCP(config.server_name)

# Instâncias globais dos componentes principais
pdf_processor: Optional[PDFProcessor] = None
vector_store: Optional[VectorStore] = None
query_handler: Optional[QueryHandler] = None


async def initialize_system() -> bool:
    """
    Inicializa todo o sistema MCP
    
    Returns:
        True se inicialização bem-sucedida, False caso contrário
    """
    global pdf_processor, vector_store, query_handler
    
    logger.info("🚀 Inicializando CV Reference MCP Server...")
    
    try:
        # Inicializar processador de PDFs
        pdf_processor = PDFProcessor()
        logger.info("✅ PDF Processor inicializado")
        
        # Inicializar Vector Store
        vector_store = VectorStore()
        
        # Tentar carregar índice existente
        if vector_store.load_index():
            logger.info("✅ Índice FAISS carregado do disco")
        else:
            # Se não existir, processar PDFs e criar índice
            logger.info("📚 Nenhum índice encontrado, processando PDFs...")
            
            # Verificar se há PDFs para processar
            pdf_files = list(config.pdfs_dir.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"⚠️ Nenhum PDF encontrado em {config.pdfs_dir}")
                logger.info("📝 Coloque os PDFs das referências no diretório 'data/pdfs/'")
                return False
            
            # Processar todos os PDFs
            chunks = pdf_processor.process_all_pdfs()
            if not chunks:
                logger.error("❌ Falha ao processar PDFs")
                return False
            
            # Construir índice FAISS
            vector_store.build_index(chunks)
            
            # Salvar índice para uso futuro
            vector_store.save_index()
            
            logger.info(f"✅ Índice FAISS criado com {len(chunks)} chunks")
        
        # Inicializar Query Handler
        query_handler = QueryHandler(vector_store)
        logger.info("✅ Query Handler inicializado")
        
        # Verificar se sistema está pronto
        if not vector_store.is_ready():
            logger.error("❌ Sistema não está pronto para uso")
            return False
        
        # Log das estatísticas do sistema
        stats = vector_store.get_stats()
        books = vector_store.list_books()
        
        logger.info(f"📊 Sistema pronto:")
        logger.info(f"  - Documentos indexados: {stats.total_documents}")
        logger.info(f"  - Dimensão dos embeddings: {stats.index_dimension}")
        logger.info(f"  - Livros disponíveis: {len(books)}")
        
        for book_code, book_info in books.items():
            logger.info(f"    • {book_info['book_title']}: {book_info['chunk_count']} chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {str(e)}")
        return False


# ============================================================================
# RECURSOS MCP (Resources)
# ============================================================================

@mcp.resource("cv-reference://search")
async def search_references() -> str:
    """
    Recurso para busca geral nas referências
    
    Returns:
        Instruções de uso da busca semântica
    """
    if not query_handler:
        return "❌ Sistema não inicializado. Execute o setup primeiro."
    
    # Estatísticas do sistema
    books = query_handler.get_book_statistics()
    total_docs = sum(book['chunk_count'] for book in books.values())
    
    instructions = f"""# 📚 Busca Semântica em Referências de CV

## Status do Sistema
- **Documentos indexados**: {total_docs}
- **Livros disponíveis**: {len(books)}

## Livros Indexados:
"""
    
    for book_code, book_info in books.items():
        instructions += f"- **{book_info['book_title']}**: {book_info['chunk_count']} chunks ({book_info['page_count']} páginas)\n"
    
    instructions += """
## Como Usar:
1. **Consulta geral**: Use a ferramenta `query_cv_references`
2. **Conceito específico**: Use o recurso `cv-reference://concept/[conceito]`
3. **Livro específico**: Adicione `book_filter` na consulta

## Exemplos de Consultas:
- "Como funciona o filtro gaussiano?"
- "Implementação do algoritmo de Canny"
- "Diferença entre SIFT e SURF"
- "Teoria matemática de convolução"
"""
    
    return instructions


@mcp.resource("cv-reference://concept/{concept}")
async def get_concept_info(concept: str) -> str:
    """
    Recurso para informações sobre conceito específico
    
    Args:
        concept: Nome do conceito a pesquisar
        
    Returns:
        Informações detalhadas sobre o conceito
    """
    if not query_handler:
        return "❌ Sistema não inicializado."
    
    # Buscar informações sobre o conceito
    response = query_handler.get_concept_info(concept)
    
    if not response:
        available_concepts = query_handler.list_available_concepts()
        return f"""❌ Conceito '{concept}' não encontrado.

📝 **Conceitos disponíveis:**
{', '.join(available_concepts[:20])}
{'...' if len(available_concepts) > 20 else ''}

💡 **Dica:** Use nomes em português como 'filtro-gaussiano', 'detecção-bordas', etc.
"""
    
    # Formatar resposta do conceito
    result_text = f"# 📖 {concept.replace('-', ' ').title()}\n\n"
    
    if response.results:
        result_text += f"**{len(response.results)} referência(s) encontrada(s)**\n\n"
        
        for i, result in enumerate(response.results[:3], 1):
            result_text += f"## {i}. {result['source']}\n"
            result_text += f"**Similaridade:** {result['similarity_score']}\n\n"
            result_text += f"{result['text'][:500]}...\n\n"
            result_text += "---\n\n"
    else:
        result_text += "❌ Nenhuma referência específica encontrada para este conceito.\n"
    
    return result_text


@mcp.resource("cv-reference://book/{book_name}")
async def get_book_info(book_name: str) -> str:
    """
    Recurso para informações sobre um livro específico
    
    Args:
        book_name: Código ou nome do livro
        
    Returns:
        Informações detalhadas sobre o livro
    """
    if not query_handler:
        return "❌ Sistema não inicializado."
    
    books = query_handler.get_book_statistics()
    
    # Tentar encontrar o livro pelo código ou nome
    book_info = None
    book_code = None
    
    for code, info in books.items():
        if (book_name.lower() == code.lower() or 
            book_name.lower() in info['book_title'].lower()):
            book_info = info
            book_code = code
            break
    
    if not book_info:
        available_books = list(books.keys())
        return f"""❌ Livro '{book_name}' não encontrado.

📚 **Livros disponíveis:**
{', '.join(available_books)}
"""
    
    # Formatar informações do livro
    info_text = f"""# 📚 {book_info['book_title']}

## Estatísticas
- **Chunks indexados**: {book_info['chunk_count']}
- **Total de palavras**: {book_info['total_words']:,}
- **Páginas processadas**: {book_info['page_count']}
- **Código do livro**: `{book_code}`

## Páginas Indexadas
{', '.join(map(str, book_info['pages'][:10]))}{'...' if len(book_info['pages']) > 10 else ''}

## Como Consultar
Use a ferramenta `query_cv_references` com `book_filter="{book_code}"` para buscar apenas neste livro.

**Exemplo:**
```
query_cv_references(
    query="algoritmo de segmentação",
    book_filter="{book_code}"
)
```
"""
    
    return info_text


# ============================================================================
# FERRAMENTAS MCP (Tools)
# ============================================================================

@mcp.tool()
async def query_cv_references(
    query: str,
    top_k: int = 5,
    book_filter: Optional[str] = None,
    similarity_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Realiza consulta semântica nas referências de CV
    
    Args:
        query: Consulta em linguagem natural
        top_k: Número máximo de resultados (1-20)
        book_filter: Filtrar por livro específico (gonzalez, szeliski, goodfellow, bishop)
        similarity_threshold: Threshold mínimo de similaridade (0.0-1.0)
        
    Returns:
        Resultados da consulta com contexto e sugestões
    """
    if not query_handler:
        return {"error": "Sistema não inicializado"}
    
    # Validar parâmetros
    top_k = max(1, min(top_k, config.max_top_k))
    similarity_threshold = max(0.0, min(similarity_threshold, 1.0))
    
    try:
        # Realizar consulta
        response = query_handler.search(
            query=query,
            top_k=top_k,
            book_filter=book_filter,
            similarity_threshold=similarity_threshold
        )
        
        # Converter para dicionário para resposta MCP
        return response.dict()
        
    except Exception as e:
        logger.error(f"Erro na consulta: {str(e)}")
        return {
            "error": f"Erro na consulta: {str(e)}",
            "query": query,
            "total_results": 0,
            "results": [],
            "context": {},
            "suggestions": []
        }


@mcp.tool()
async def list_available_books() -> Dict[str, Any]:
    """
    Lista todos os livros disponíveis para consulta
    
    Returns:
        Informações sobre os livros indexados
    """
    if not query_handler:
        return {"error": "Sistema não inicializado"}
    
    try:
        books_stats = query_handler.get_book_statistics()
        
        return {
            "total_books": len(books_stats),
            "books": books_stats,
            "book_codes": list(books_stats.keys()),
            "total_documents": sum(book['chunk_count'] for book in books_stats.values())
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar livros: {str(e)}")
        return {"error": f"Erro ao listar livros: {str(e)}"}


@mcp.tool()
async def get_system_stats() -> Dict[str, Any]:
    """
    Obtém estatísticas detalhadas do sistema
    
    Returns:
        Estatísticas completas do MCP server
    """
    if not vector_store or not query_handler:
        return {"error": "Sistema não inicializado"}
    
    try:
        vector_stats = vector_store.get_stats()
        books_stats = query_handler.get_book_statistics()
        
        # Calcular estatísticas gerais
        total_words = sum(book['total_words'] for book in books_stats.values())
        total_chunks = sum(book['chunk_count'] for book in books_stats.values())
        
        return {
            "server_info": {
                "name": config.server_name,
                "version": config.server_version,
                "description": config.server_description,
            },
            "vector_store": {
                "model_name": vector_stats.model_name,
                "total_documents": vector_stats.total_documents,
                "index_dimension": vector_stats.index_dimension,
                "index_type": vector_stats.index_type,
            },
            "content": {
                "total_books": len(books_stats),
                "total_chunks": total_chunks,
                "total_words": total_words,
                "books": books_stats
            },
            "config": {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "default_top_k": config.default_top_k,
                "similarity_threshold": config.similarity_threshold,
            }
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {str(e)}")
        return {"error": f"Erro ao obter estatísticas: {str(e)}"}


@mcp.tool()
async def list_cv_concepts() -> Dict[str, Any]:
    """
    Lista conceitos de CV disponíveis para consulta
    
    Returns:
        Lista de conceitos organizados por categoria
    """
    if not query_handler:
        return {"error": "Sistema não inicializado"}
    
    try:
        concepts = query_handler.list_available_concepts()
        
        # Organizar conceitos por categoria
        categories = {
            "Fundamentos": [c for c in concepts if any(term in c for term in ['imagem', 'pixel', 'rgb', 'hsv', 'cinza'])],
            "Filtros": [c for c in concepts if 'filtro' in c or any(term in c for term in ['gaussiano', 'sobel', 'laplaciano'])],
            "Segmentação": [c for c in concepts if any(term in c for term in ['segmentação', 'limiarização', 'otsu', 'canny', 'hough'])],
            "Morfologia": [c for c in concepts if any(term in c for term in ['erosão', 'dilatação', 'abertura', 'fechamento', 'morfologia'])],
            "Características": [c for c in concepts if any(term in c for term in ['sift', 'surf', 'orb', 'hog', 'lbp'])],
            "Deep Learning": [c for c in concepts if any(term in c for term in ['cnn', 'yolo', 'rcnn', 'unet'])],
        }
        
        # Remover duplicatas e conceitos não categorizados
        categorized = set()
        for cat_concepts in categories.values():
            categorized.update(cat_concepts)
        
        uncategorized = [c for c in concepts if c not in categorized]
        if uncategorized:
            categories["Outros"] = uncategorized
        
        return {
            "total_concepts": len(concepts),
            "categories": categories,
            "all_concepts": sorted(concepts)
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar conceitos: {str(e)}")
        return {"error": f"Erro ao listar conceitos: {str(e)}"}


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

async def main():
    """Função principal do servidor MCP"""
    logger.info(f"🔧 Iniciando {config.server_name} v{config.server_version}")
    
    # Inicializar sistema
    success = await initialize_system()
    
    if not success:
        logger.error("❌ Falha na inicialização do sistema")
        sys.exit(1)
    
    logger.info("✅ CV Reference MCP Server pronto!")
    logger.info("📡 Aguardando conexões MCP...")
    
    # O FastMCP cuida do resto
    return mcp


def run_server():
    """Executa o servidor MCP"""
    try:
        app = asyncio.run(main())
        return app
    except KeyboardInterrupt:
        logger.info("🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal no servidor: {str(e)}")
        sys.exit(1)


# Instância do app para uso externo
app = mcp

if __name__ == "__main__":
    run_server()