#!/usr/bin/env python3
"""
Script de Teste para CV Reference MCP Server

Este script testa as principais funcionalidades do servidor MCP,
incluindo processamento de PDFs, vetorização e consultas semânticas.

Uso:
    python scripts/test_server.py [--verbose]
"""

import argparse
import sys
import asyncio
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import config
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from query_handler import QueryHandler


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPTester:
    """Classe para testar funcionalidades do MCP"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.pdf_processor = None
        self.vector_store = None
        self.query_handler = None
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def test_config(self) -> bool:
        """Testa configurações básicas"""
        logger.info("🔧 Testando configurações...")
        
        try:
            # Verificar paths
            assert config.data_dir.exists(), f"Diretório de dados não existe: {config.data_dir}"
            assert config.pdfs_dir.exists(), f"Diretório de PDFs não existe: {config.pdfs_dir}"
            
            # Verificar configurações de embedding
            assert config.embedding_model, "Modelo de embedding não configurado"
            assert config.embedding_dimension > 0, "Dimensão de embedding inválida"
            
            # Verificar configurações de chunk
            assert config.chunk_size > 0, "Tamanho de chunk inválido"
            assert config.chunk_overlap >= 0, "Sobreposição de chunk inválida"
            
            logger.info("✅ Configurações OK")
            return True
            
        except AssertionError as e:
            logger.error(f"❌ Erro na configuração: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro inesperado na configuração: {e}")
            return False

    def test_pdf_processing(self) -> bool:
        """Testa processamento de PDFs"""
        logger.info("📚 Testando processamento de PDFs...")
        
        try:
            self.pdf_processor = PDFProcessor()
            
            # Verificar se há PDFs para testar
            pdf_files = list(config.pdfs_dir.glob("*.pdf"))
            if not pdf_files:
                logger.warning("⚠️ Nenhum PDF encontrado para teste")
                return True  # Não é erro, apenas sem dados
            
            logger.info(f"📁 Encontrados {len(pdf_files)} PDFs")
            
            # Testar com primeiro PDF
            first_pdf = pdf_files[0]
            logger.info(f"📖 Testando com: {first_pdf.name}")
            
            book_code = self.pdf_processor._detect_book_code(first_pdf.stem)
            chunks = self.pdf_processor.extract_text_from_pdf(first_pdf, book_code)
            
            assert len(chunks) > 0, f"Nenhum chunk extraído de {first_pdf.name}"
            
            # Verificar estrutura dos chunks
            first_chunk = chunks[0]
            assert first_chunk.text, "Chunk sem texto"
            assert first_chunk.page_number > 0, "Número de página inválido"
            assert first_chunk.chunk_id, "ID do chunk vazio"
            assert first_chunk.book_code, "Código do livro vazio"
            
            logger.info(f"✅ PDF processado: {len(chunks)} chunks extraídos")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento de PDF: {e}")
            return False

    def test_vector_store(self) -> bool:
        """Testa sistema de vetorização"""
        logger.info("🔍 Testando vector store...")
        
        try:
            self.vector_store = VectorStore()
            
            # Tentar carregar índice existente
            if self.vector_store.load_index():
                logger.info("📁 Índice carregado do disco")
                
                # Verificar integridade
                stats = self.vector_store.get_stats()
                assert stats.total_documents > 0, "Índice vazio"
                assert stats.index_dimension > 0, "Dimensão inválida"
                
                logger.info(f"✅ Índice válido: {stats.total_documents} docs")
                return True
                
            else:
                logger.info("📚 Criando índice de teste...")
                
                # Criar alguns chunks de teste se não há PDFs processados
                if not self.pdf_processor:
                    logger.warning("⚠️ PDF processor não inicializado, pulando teste de vector store")
                    return True
                
                # Processar um PDF pequeno para teste
                pdf_files = list(config.pdfs_dir.glob("*.pdf"))
                if not pdf_files:
                    logger.warning("⚠️ Nenhum PDF para testar vector store")
                    return True
                
                # Usar chunks já processados ou processar primeiro PDF
                chunks = self.pdf_processor.load_processed_chunks()
                if not chunks:
                    first_pdf = pdf_files[0]
                    book_code = self.pdf_processor._detect_book_code(first_pdf.stem)
                    chunks = self.pdf_processor.extract_text_from_pdf(first_pdf, book_code)[:10]  # Apenas 10 chunks para teste
                
                if chunks:
                    # Construir índice de teste
                    self.vector_store.build_index(chunks[:5])  # Apenas alguns chunks para teste
                    
                    stats = self.vector_store.get_stats()
                    assert stats.total_documents > 0, "Índice de teste vazio"
                    
                    logger.info(f"✅ Vector store testado: {stats.total_documents} docs")
                    return True
                else:
                    logger.warning("⚠️ Sem chunks para testar vector store")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Erro no vector store: {e}")
            return False

    def test_query_handler(self) -> bool:
        """Testa handler de consultas"""
        logger.info("❓ Testando query handler...")
        
        try:
            if not self.vector_store:
                logger.warning("⚠️ Vector store não inicializado, pulando teste de query handler")
                return True
            
            if not self.vector_store.is_ready():
                logger.warning("⚠️ Vector store não está pronto, pulando teste de query handler")
                return True
            
            self.query_handler = QueryHandler(self.vector_store)
            
            # Testar consultas básicas
            test_queries = [
                "filtro gaussiano",
                "detecção de bordas", 
                "segmentação",
                "o que é convolução"
            ]
            
            for query in test_queries:
                logger.info(f"🔍 Testando query: '{query}'")
                
                # Processar contexto da query
                context = self.query_handler.process_query(query)
                assert context.original_query == query, "Query original não preservada"
                assert context.processed_query, "Query processada vazia"
                
                # Fazer busca
                response = self.query_handler.search(query, top_k=3)
                assert response.query == query, "Query na resposta incorreta"
                assert response.total_results >= 0, "Número de resultados inválido"
                
                if response.total_results > 0:
                    logger.info(f"  ✅ {response.total_results} resultado(s) encontrado(s)")
                    
                    # Verificar primeiro resultado
                    first_result = response.results[0]
                    assert first_result['text'], "Resultado sem texto"
                    assert first_result['similarity_score'] >= 0, "Score de similaridade inválido"
                    assert first_result['book_code'], "Código do livro vazio"
                    
                else:
                    logger.info(f"  ⚠️ Nenhum resultado (pode ser normal para teste pequeno)")
            
            logger.info("✅ Query handler funcionando corretamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no query handler: {e}")
            return False

    def test_integration(self) -> bool:
        """Testa integração completa"""
        logger.info("🔗 Testando integração completa...")
        
        try:
            if not all([self.pdf_processor, self.vector_store, self.query_handler]):
                logger.warning("⚠️ Nem todos os componentes foram inicializados")
                return True
            
            # Teste de fluxo completo
            logger.info("📊 Obtendo estatísticas do sistema...")
            
            if self.vector_store.is_ready():
                # Estatísticas do vector store
                vector_stats = self.vector_store.get_stats()
                books_stats = self.query_handler.get_book_statistics()
                
                logger.info(f"📈 Estatísticas:")
                logger.info(f"  • Documentos: {vector_stats.total_documents}")
                logger.info(f"  • Livros: {len(books_stats)}")
                logger.info(f"  • Modelo: {vector_stats.model_name}")
                
                # Testar listagem de conceitos
                concepts = self.query_handler.list_available_concepts()
                logger.info(f"  • Conceitos disponíveis: {len(concepts)}")
                
                # Testar busca por conceito específico
                if concepts:
                    concept = concepts[0]
                    concept_info = self.query_handler.get_concept_info(concept)
                    if concept_info:
                        logger.info(f"  • Teste de conceito '{concept}': OK")
                
                logger.info("✅ Integração completa funcionando")
            else:
                logger.info("ℹ️ Sistema não está completamente pronto (normal para teste)")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na integração: {e}")
            return False

    async def run_all_tests(self) -> bool:
        """Executa todos os testes"""
        logger.info("🧪 CV Reference MCP - Suite de Testes")
        logger.info("=" * 50)
        
        tests = [
            ("Configuração", self.test_config),
            ("Processamento PDF", self.test_pdf_processing),
            ("Vector Store", self.test_vector_store),
            ("Query Handler", self.test_query_handler),
            ("Integração", self.test_integration),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Executando teste: {test_name}")
            logger.info("-" * 30)
            
            try:
                result = test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name}: PASSOU")
                else:
                    logger.error(f"❌ {test_name}: FALHOU")
                    
            except Exception as e:
                logger.error(f"💥 {test_name}: ERRO - {e}")
                results[test_name] = False
        
        # Resumo final
        logger.info("\n📊 Resumo dos Testes")
        logger.info("=" * 30)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSOU" if result else "❌ FALHOU"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nResultado Final: {passed}/{total} testes passaram")
        
        if passed == total:
            logger.info("🎉 Todos os testes passaram!")
            return True
        else:
            logger.error(f"❌ {total - passed} teste(s) falharam")
            return False


async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Suite de testes para CV Reference MCP Server"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar log verboso"
    )
    
    args = parser.parse_args()
    
    tester = MCPTester(verbose=args.verbose)
    success = await tester.run_all_tests()
    
    if success:
        logger.info("\n🎯 Sistema pronto para uso!")
        logger.info("💡 Para executar o servidor MCP:")
        logger.info("   python -m src.server")
    else:
        logger.error("\n❌ Alguns testes falharam")
        logger.info("💡 Verifique os erros acima e tente novamente")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())