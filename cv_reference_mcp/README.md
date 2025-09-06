# 📚 CV Reference MCP Server

Um servidor MCP (Model Context Protocol) para consulta semântica de referências bibliográficas de Visão Computacional usando RAG (Retrieval-Augmented Generation).

## 🎯 Objetivo

Este servidor permite consultar semanticamente os conteúdos dos principais livros de referência em Visão Computacional, fornecendo contexto inteligente durante o desenvolvimento e estudo de algoritmos de CV.

### 📖 Livros Suportados
- **Digital Image Processing** - Gonzalez & Woods
- **Computer Vision: Algorithms and Applications** - Szeliski  
- **Deep Learning** - Goodfellow, Bengio & Courville
- **Pattern Recognition and Machine Learning** - Bishop

## 🏗️ Arquitetura

```
cv_reference_mcp/
├── src/                    # Código fonte principal
│   ├── server.py          # Servidor MCP principal (FastMCP)
│   ├── pdf_processor.py   # Extração e chunking de PDFs
│   ├── vector_store.py    # FAISS + sentence-transformers
│   ├── query_handler.py   # Processamento inteligente de consultas
│   └── config.py          # Configurações centralizadas
├── data/                   # Dados do sistema
│   ├── pdfs/              # PDFs das referências (você adiciona)
│   ├── processed/         # Textos extraídos e processados
│   └── vectors/           # Índices FAISS salvos
├── scripts/               # Scripts utilitários
│   ├── setup_vectors.py   # Setup inicial de vetorização
│   └── test_server.py     # Testes do sistema
└── requirements.txt       # Dependências
```

## 🚀 Instalação e Setup

### 1. Instalar Dependências

```bash
# Via pip
pip install -r requirements.txt

# Ou via uv (recomendado)
uv add -r requirements.txt
```

### 2. Adicionar PDFs das Referências

Coloque os PDFs dos livros de referência no diretório `data/pdfs/`:

```bash
data/pdfs/
├── gonzalez_woods_digital_image_processing.pdf
├── szeliski_computer_vision.pdf
├── goodfellow_deep_learning.pdf
└── bishop_pattern_recognition.pdf
```

> **Nota:** Os nomes dos arquivos são detectados automaticamente. Use nomes descritivos contendo o autor ou título.

### 3. Executar Setup de Vetorização

```bash
# Setup inicial (processa PDFs e cria índice FAISS)
python scripts/setup_vectors.py

# Forçar recriação do índice
python scripts/setup_vectors.py --force-rebuild
```

### 4. Testar o Sistema

```bash
# Executar testes
python scripts/test_server.py

# Testes verbosos
python scripts/test_server.py --verbose
```

### 5. Executar o Servidor MCP

```bash
# Iniciar servidor
python -m src.server

# Ou usando uv
uv run -m src.server
```

## 🔧 Configuração no Claude Code

Para usar com Claude Code, adicione ao arquivo de configuração MCP:

```json
{
  "mcpServers": {
    "cv-reference": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/cv_reference_mcp",
      "env": {
        "PYTHONPATH": "/path/to/cv_reference_mcp/src"
      }
    }
  }
}
```

## 📡 Recursos e Ferramentas MCP

### 🔍 Recursos (Resources)

- **`cv-reference://search`**: Instruções gerais de busca
- **`cv-reference://concept/{conceito}`**: Info sobre conceito específico  
- **`cv-reference://book/{livro}`**: Informações sobre livro específico

### 🛠️ Ferramentas (Tools)

- **`query_cv_references()`**: Consulta semântica principal
- **`list_available_books()`**: Lista livros indexados
- **`get_system_stats()`**: Estatísticas do sistema
- **`list_cv_concepts()`**: Conceitos de CV disponíveis

## 💡 Exemplos de Uso

### Consulta Básica
```python
# Via MCP tool
query_cv_references(
    query="Como funciona o filtro gaussiano?",
    top_k=5
)
```

### Busca em Livro Específico
```python
query_cv_references(
    query="implementação de convolução",
    book_filter="gonzalez",
    top_k=3
)
```

### Conceito Específico
```python
# Via MCP resource
cv-reference://concept/detecção-bordas
```

## 🧠 Funcionalidades Inteligentes

### 🔎 Análise de Consultas
- **Detecção de tipo**: Conceito, algoritmo, implementação, teoria, comparação
- **Expansão semântica**: Sinônimos e termos relacionados de CV
- **Sugestão de livros**: Recomenda fonte mais relevante automaticamente

### 📚 Conceitos Suportados
- **Fundamentos**: RGB, HSV, amostragem, quantização
- **Processamento**: Convolução, filtros (Gaussiano, Sobel, Laplaciano)
- **Segmentação**: Limiarização, Otsu, Canny, Hough, Watershed
- **Morfologia**: Erosão, dilatação, abertura, fechamento
- **Características**: SIFT, SURF, ORB, HOG, LBP
- **Deep Learning**: CNN, YOLO, R-CNN, U-Net, Dropout

### 🎯 Busca Contextual
- **Threshold adaptativo**: Ajusta similaridade baseado no contexto
- **Filtros inteligentes**: Sugere livro mais relevante automaticamente  
- **Ranking semântico**: Ordena por relevância conceitual

## 📊 Configurações

### Parâmetros de Embeddings
- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensão**: 384
- **Tipo de índice**: FAISS IndexFlatIP (Inner Product)

### Parâmetros de Chunking
- **Tamanho do chunk**: 512 caracteres
- **Sobreposição**: 50 caracteres  
- **Tamanho mínimo**: 100 caracteres

### Parâmetros de Busca
- **Top-K padrão**: 5 resultados
- **Máximo**: 20 resultados
- **Threshold de similaridade**: 0.3

## 🔧 Desenvolvimento

### Estrutura do Código

**`config.py`**: Configurações centralizadas e mapeamentos de conceitos

**`pdf_processor.py`**: 
- Extração de texto com PyMuPDF
- Limpeza e pré-processamento
- Chunking inteligente com preservação de contexto

**`vector_store.py`**:
- Embeddings com sentence-transformers
- Indexação FAISS otimizada
- Persistência e carregamento

**`query_handler.py`**:
- Análise e processamento de consultas
- Expansão semântica automática
- Formatação de respostas

**`server.py`**:
- Servidor FastMCP
- Recursos e ferramentas MCP
- Inicialização e orquestração

### Adicionando Novos Conceitos

Edite `config.py` e adicione ao array `cv_concepts`:

```python
cv_concepts: List[str] = [
    # ... conceitos existentes
    "novo-conceito",
    "outro-conceito-cv"
]
```

### Adicionando Sinônimos

Edite `query_handler.py` no dicionário `cv_synonyms`:

```python
cv_synonyms = {
    # ... sinônimos existentes
    "novo-termo": ["sinonimo1", "sinonimo2", "termo-relacionado"]
}
```

## 🧪 Testes

### Suite Completa
```bash
python scripts/test_server.py
```

### Componentes Individuais
```bash
# Testar apenas PDF processing
python -c "from src.pdf_processor import PDFProcessor; p = PDFProcessor(); print('OK')"

# Testar apenas Vector Store  
python -c "from src.vector_store import VectorStore; v = VectorStore(); print('OK')"
```

## 🐛 Troubleshooting

### Problemas Comuns

**"Nenhum PDF encontrado"**
- Verifique se os PDFs estão em `data/pdfs/`
- Certifique-se que têm extensão `.pdf`

**"Erro ao carregar modelo"**
- Verifique conexão com internet (primeiro download)
- Instale `sentence-transformers` corretamente

**"Índice FAISS corrompido"**
```bash
# Recriar índice
python scripts/setup_vectors.py --force-rebuild
```

**"Sem resultados na busca"**
- Reduza `similarity_threshold`
- Tente consultas mais genéricas
- Verifique se o índice foi criado corretamente

### Logs e Debug

```bash
# Log detalhado
PYTHONPATH=src python -m server --verbose

# Testar com debug
python scripts/test_server.py --verbose
```

## 📈 Estatísticas de Performance

### Benchmarks Típicos
- **Processamento**: ~2-5 páginas/segundo
- **Indexação**: ~50-100 chunks/segundo  
- **Busca**: <100ms para consultas típicas
- **Memória**: ~200-500MB (dependendo do tamanho do corpus)

### Otimizações
- Embeddings normalizados para cosine similarity eficiente
- Índice FAISS IndexFlatIP otimizado  
- Chunking inteligente preservando contexto
- Cache de consultas frequentes

## 📄 Licença

Este projeto é parte dos estudos de Visão Computacional e está disponível sob licença MIT.

## 🤝 Contribuição

Para melhorias e sugestões:
1. Teste thoroughly com `scripts/test_server.py`
2. Mantenha documentação atualizada
3. Siga padrões de código existentes
4. Adicione testes para novas funcionalidades

---

**💡 Dica:** Este MCP server foi projetado especificamente para apoiar o aprendizado pedagógico de Visão Computacional, priorizando clareza e compreensão sobre performance bruta.