"""
Processamento e Extração de PDFs

Este módulo é responsável por extrair, limpar e estruturar o conteúdo de PDFs
das referências bibliográficas, preparando-os para vetorização e busca semântica.

Principais funcionalidades:
- Extração de texto usando PyMuPDF
- Limpeza e pré-processamento de texto
- Chunking inteligente com sobreposição
- Preservação de metadados (página, seção, livro)
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

import fitz  # PyMuPDF
from pydantic import BaseModel

from config import config


# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Representa um chunk de texto extraído de um PDF"""
    text: str
    page_number: int
    chunk_id: str
    book_code: str
    book_title: str
    start_char: int
    end_char: int
    word_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte o chunk para dicionário"""
        return asdict(self)


class ProcessingStats(BaseModel):
    """Estatísticas do processamento de PDFs"""
    total_pdfs: int = 0
    successful_pdfs: int = 0
    failed_pdfs: int = 0
    total_pages: int = 0
    total_chunks: int = 0
    total_words: int = 0
    errors: List[str] = []


class PDFProcessor:
    """Processador de PDFs para extração e chunking de texto"""
    
    def __init__(self):
        self.config = config
        self.stats = ProcessingStats()
        
        # Padrões regex para limpeza de texto
        self.cleanup_patterns = [
            (r'\s+', ' '),  # Múltiplos espaços -> espaço único
            (r'\n\s*\n', '\n\n'),  # Múltiplas quebras -> dupla quebra
            (r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\+\=\<\>\@\#\$\%\&\*\/\\]', ''),  # Remove chars especiais
            (r'\s*\n\s*', '\n'),  # Limpa quebras de linha
        ]
        
        # Padrões para detectar headers/footers
        self.header_footer_patterns = [
            r'^\d+\s*$',  # Só números (páginas)
            r'^Chapter\s+\d+',  # Capítulos
            r'^CHAPTER\s+\d+',
            r'^\d+\.\d+\s+[A-Z]',  # Numeração de seções
            r'^Figure\s+\d+',  # Figuras
            r'^Table\s+\d+',  # Tabelas
            r'^\s*©.*\d{4}',  # Copyright
        ]

    def extract_text_from_pdf(self, pdf_path: Path, book_code: str) -> List[DocumentChunk]:
        """
        Extrai texto de um PDF e retorna lista de chunks processados
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            book_code: Código identificador do livro
            
        Returns:
            Lista de DocumentChunk com o texto extraído e processado
        """
        logger.info(f"Processando PDF: {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
            book_title = config.get_book_name(book_code)
            all_chunks = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Limpar e pré-processar o texto da página
                cleaned_text = self._clean_text(text)
                
                if len(cleaned_text.strip()) < config.min_chunk_size:
                    continue  # Pula páginas muito pequenas
                
                # Criar chunks da página
                page_chunks = self._create_chunks(
                    text=cleaned_text,
                    page_number=page_num + 1,  # Página começa em 1
                    book_code=book_code,
                    book_title=book_title
                )
                
                all_chunks.extend(page_chunks)
                self.stats.total_pages += 1
            
            doc.close()
            
            self.stats.successful_pdfs += 1
            self.stats.total_chunks += len(all_chunks)
            self.stats.total_words += sum(chunk.word_count for chunk in all_chunks)
            
            logger.info(f"✅ PDF processado: {len(all_chunks)} chunks extraídos")
            return all_chunks
            
        except Exception as e:
            error_msg = f"Erro ao processar {pdf_path.name}: {str(e)}"
            logger.error(error_msg)
            self.stats.failed_pdfs += 1
            self.stats.errors.append(error_msg)
            return []

    def _clean_text(self, text: str) -> str:
        """
        Limpa e pré-processa texto extraído do PDF
        
        Args:
            text: Texto bruto extraído
            
        Returns:
            Texto limpo e pré-processado
        """
        if not text:
            return ""
        
        # Aplicar padrões de limpeza
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Remover possíveis headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Pular linhas muito curtas ou que parecem headers/footers
            if len(line) < 10:
                continue
            
            # Verificar se é header/footer
            is_header_footer = any(
                re.match(pattern, line, re.IGNORECASE) 
                for pattern in self.header_footer_patterns
            )
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()

    def _create_chunks(
        self, 
        text: str, 
        page_number: int, 
        book_code: str, 
        book_title: str
    ) -> List[DocumentChunk]:
        """
        Divide texto em chunks com sobreposição
        
        Args:
            text: Texto limpo da página
            page_number: Número da página
            book_code: Código do livro
            book_title: Título completo do livro
            
        Returns:
            Lista de chunks da página
        """
        chunks = []
        text_length = len(text)
        
        # Se o texto é menor que chunk_size, criar um único chunk
        if text_length <= config.chunk_size:
            if text_length >= config.min_chunk_size:
                chunk_id = f"{book_code}_p{page_number}_c1"
                chunk = DocumentChunk(
                    text=text,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    book_code=book_code,
                    book_title=book_title,
                    start_char=0,
                    end_char=text_length,
                    word_count=len(text.split())
                )
                chunks.append(chunk)
            return chunks
        
        # Dividir em múltiplos chunks com sobreposição
        chunk_num = 1
        start = 0
        
        while start < text_length:
            end = min(start + config.chunk_size, text_length)
            
            # Tentar quebrar em uma fronteira de palavra
            if end < text_length:
                # Procurar por um espaço próximo ao fim do chunk
                space_pos = text.rfind(' ', start + config.chunk_size - 100, end)
                if space_pos > start:
                    end = space_pos
            
            chunk_text = text[start:end].strip()
            
            # Só criar chunk se tiver tamanho mínimo
            if len(chunk_text) >= config.min_chunk_size:
                chunk_id = f"{book_code}_p{page_number}_c{chunk_num}"
                chunk = DocumentChunk(
                    text=chunk_text,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    book_code=book_code,
                    book_title=book_title,
                    start_char=start,
                    end_char=end,
                    word_count=len(chunk_text.split())
                )
                chunks.append(chunk)
                chunk_num += 1
            
            # Próximo chunk com sobreposição
            start = end - config.chunk_overlap
            
            # Evitar loops infinitos
            if start <= end - config.chunk_size:
                start = end
        
        return chunks

    def process_all_pdfs(self) -> List[DocumentChunk]:
        """
        Processa todos os PDFs no diretório de PDFs
        
        Returns:
            Lista completa de chunks de todos os PDFs
        """
        logger.info(f"Iniciando processamento de PDFs em: {config.pdfs_dir}")
        
        self.stats = ProcessingStats()  # Reset das estatísticas
        all_chunks = []
        
        # Buscar todos os PDFs
        pdf_files = list(config.pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"Nenhum PDF encontrado em {config.pdfs_dir}")
            return all_chunks
        
        self.stats.total_pdfs = len(pdf_files)
        logger.info(f"Encontrados {len(pdf_files)} PDFs para processar")
        
        # Processar cada PDF
        for pdf_path in pdf_files:
            # Tentar detectar o código do livro pelo nome do arquivo
            book_code = self._detect_book_code(pdf_path.stem)
            
            # Processar PDF
            pdf_chunks = self.extract_text_from_pdf(pdf_path, book_code)
            all_chunks.extend(pdf_chunks)
        
        # Salvar chunks processados
        self._save_processed_chunks(all_chunks)
        
        # Log das estatísticas finais
        logger.info(f"Processamento concluído:")
        logger.info(f"  PDFs processados: {self.stats.successful_pdfs}/{self.stats.total_pdfs}")
        logger.info(f"  Total de páginas: {self.stats.total_pages}")
        logger.info(f"  Total de chunks: {self.stats.total_chunks}")
        logger.info(f"  Total de palavras: {self.stats.total_words}")
        
        if self.stats.errors:
            logger.warning(f"  Erros encontrados: {len(self.stats.errors)}")
            for error in self.stats.errors:
                logger.warning(f"    - {error}")
        
        return all_chunks

    def _detect_book_code(self, filename: str) -> str:
        """
        Detecta o código do livro baseado no nome do arquivo
        
        Args:
            filename: Nome do arquivo (sem extensão)
            
        Returns:
            Código do livro detectado ou 'unknown'
        """
        filename_lower = filename.lower()
        
        # Mapear padrões comuns nos nomes de arquivos
        patterns = {
            'gonzalez': ['gonzalez', 'woods', 'digital_image_processing', 'dip'],
            'szeliski': ['szeliski', 'computer_vision', 'algorithms_applications'],
            'goodfellow': ['goodfellow', 'deep_learning', 'bengio', 'courville'],
            'bishop': ['bishop', 'pattern_recognition', 'machine_learning', 'prml'],
        }
        
        for book_code, keywords in patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return book_code
        
        return 'unknown'

    def _save_processed_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Salva chunks processados em arquivo JSON
        
        Args:
            chunks: Lista de chunks para salvar
        """
        if not chunks:
            return
        
        output_file = config.processed_dir / "processed_chunks.json"
        
        # Converter chunks para dicionários
        chunks_data = [chunk.to_dict() for chunk in chunks]
        
        # Adicionar metadados
        data = {
            "metadata": {
                "total_chunks": len(chunks),
                "processing_stats": self.stats.dict(),
                "config": {
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "min_chunk_size": config.min_chunk_size,
                }
            },
            "chunks": chunks_data
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Chunks salvos em: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar chunks: {str(e)}")

    def load_processed_chunks(self) -> List[DocumentChunk]:
        """
        Carrega chunks previamente processados
        
        Returns:
            Lista de chunks carregados ou lista vazia se não existir
        """
        chunks_file = config.processed_dir / "processed_chunks.json"
        
        if not chunks_file.exists():
            logger.info("Nenhum arquivo de chunks processados encontrado")
            return []
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Converter dicionários de volta para objetos DocumentChunk
            chunks = [
                DocumentChunk(**chunk_data) 
                for chunk_data in data.get("chunks", [])
            ]
            
            logger.info(f"✅ Carregados {len(chunks)} chunks processados")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar chunks processados: {str(e)}")
            return []

    def get_stats(self) -> ProcessingStats:
        """Retorna estatísticas do último processamento"""
        return self.stats