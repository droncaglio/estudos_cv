#!/usr/bin/env python3
"""
Script de Setup para Vetorização de PDFs

Este script processa os PDFs das referências bibliográficas e cria o índice
FAISS para busca semântica. Execute este script após adicionar novos PDFs
ao diretório data/pdfs/.

Uso:
    python scripts/setup_vectors.py [--force-rebuild]
"""

import argparse
import sys
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import config
from pdf_processor import PDFProcessor
from vector_store import VectorStore


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites() -> bool:
    """
    Verifica se os pré-requisitos estão atendidos
    
    Returns:
        True se tudo ok, False caso contrário
    """
    logger.info("🔍 Verificando pré-requisitos...")
    
    # Verificar se diretório de PDFs existe e tem conteúdo
    if not config.pdfs_dir.exists():
        logger.error(f"❌ Diretório de PDFs não encontrado: {config.pdfs_dir}")
        logger.info(f"💡 Crie o diretório e adicione os PDFs das referências")
        return False
    
    pdf_files = list(config.pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"❌ Nenhum PDF encontrado em: {config.pdfs_dir}")
        logger.info("💡 Adicione os PDFs das referências bibliográficas:")
        logger.info("   • Digital Image Processing - Gonzalez & Woods")
        logger.info("   • Computer Vision - Szeliski")
        logger.info("   • Deep Learning - Goodfellow, Bengio & Courville")
        logger.info("   • Pattern Recognition and Machine Learning - Bishop")
        return False
    
    logger.info(f"✅ Encontrados {len(pdf_files)} PDFs:")
    for pdf_file in pdf_files:
        logger.info(f"   • {pdf_file.name}")
    
    return True


def setup_vectors(force_rebuild: bool = False) -> bool:
    """
    Configura o sistema de vetorização
    
    Args:
        force_rebuild: Se True, reconstrói mesmo se já existir
        
    Returns:
        True se sucesso, False caso contrário
    """
    try:
        logger.info("🚀 Iniciando setup de vetorização...")
        
        # Verificar se já existe índice
        if config.faiss_index_path.exists() and not force_rebuild:
            logger.info("📁 Índice FAISS já existe")
            
            # Tentar carregar para verificar integridade
            vector_store = VectorStore()
            if vector_store.load_index():
                stats = vector_store.get_stats()
                books = vector_store.list_books()
                
                logger.info("✅ Índice carregado com sucesso:")
                logger.info(f"   • Documentos: {stats.total_documents}")
                logger.info(f"   • Livros: {len(books)}")
                logger.info("💡 Use --force-rebuild para recriar o índice")
                return True
            else:
                logger.warning("⚠️ Índice existente corrompido, recriando...")
                force_rebuild = True
        
        # Inicializar processador de PDFs
        logger.info("📚 Inicializando processador de PDFs...")
        pdf_processor = PDFProcessor()
        
        # Processar todos os PDFs
        logger.info("📖 Processando PDFs...")
        chunks = pdf_processor.process_all_pdfs()
        
        if not chunks:
            logger.error("❌ Nenhum chunk foi gerado dos PDFs")
            return False
        
        # Estatísticas do processamento
        stats = pdf_processor.get_stats()
        logger.info(f"📊 Processamento concluído:")
        logger.info(f"   • PDFs processados: {stats.successful_pdfs}/{stats.total_pdfs}")
        logger.info(f"   • Páginas: {stats.total_pages}")
        logger.info(f"   • Chunks: {stats.total_chunks}")
        logger.info(f"   • Palavras: {stats.total_words:,}")
        
        if stats.errors:
            logger.warning(f"⚠️ {len(stats.errors)} erro(s) encontrado(s):")
            for error in stats.errors:
                logger.warning(f"   • {error}")
        
        # Inicializar Vector Store e criar índice
        logger.info("🔧 Criando índice FAISS...")
        vector_store = VectorStore()
        vector_store.build_index(chunks)
        
        # Salvar índice
        logger.info("💾 Salvando índice...")
        vector_store.save_index()
        
        # Estatísticas finais
        vector_stats = vector_store.get_stats()
        books_stats = vector_store.list_books()
        
        logger.info("✅ Setup de vetorização concluído!")
        logger.info(f"📈 Estatísticas finais:")
        logger.info(f"   • Modelo: {vector_stats.model_name}")
        logger.info(f"   • Dimensão: {vector_stats.index_dimension}")
        logger.info(f"   • Documentos indexados: {vector_stats.total_documents}")
        logger.info(f"   • Livros disponíveis: {len(books_stats)}")
        
        logger.info("📚 Livros indexados:")
        for book_code, book_info in books_stats.items():
            logger.info(f"   • {book_info['book_title']}: {book_info['chunk_count']} chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no setup: {str(e)}")
        return False


def main():
    """Função principal do script"""
    parser = argparse.ArgumentParser(
        description="Setup de vetorização para CV Reference MCP Server"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Forçar recriação do índice mesmo se já existir"
    )
    
    args = parser.parse_args()
    
    logger.info("🔍 CV Reference MCP - Setup de Vetorização")
    logger.info("=" * 50)
    
    # Verificar pré-requisitos
    if not check_prerequisites():
        logger.error("❌ Pré-requisitos não atendidos")
        sys.exit(1)
    
    # Configurar vetorização
    success = setup_vectors(force_rebuild=args.force_rebuild)
    
    if success:
        logger.info("🎉 Setup concluído com sucesso!")
        logger.info("💡 Agora você pode executar o servidor MCP:")
        logger.info("   python -m src.server")
    else:
        logger.error("❌ Setup falhou")
        sys.exit(1)


if __name__ == "__main__":
    main()