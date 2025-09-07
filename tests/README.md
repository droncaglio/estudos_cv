# 🧪 Testes Organizados por Tópico

Esta pasta contém todos os testes e experimentos do projeto, organizados pedagogicamente seguindo a ementa de estudos.

## 📁 Estrutura dos Testes

### 01_fundamentos/
**Fundamentos de Processamento de Imagens**
- `conversoes_cinza/` - Testes de conversão RGB para escala de cinza
- `operacoes_pontuais/` - Brilho, contraste, correção gama
- `histogramas/` - Equalização e análise de histogramas

### 02_filtragem/
**Filtragem de Imagens**
- `espacial/` - Convolução, suavização, realce de bordas
- `frequencia/` - Transformada de Fourier, filtros passa-baixa/alta

### 03_segmentacao/
**Segmentação e Detecção**
- `limiarizacao/` - Limiarização global e adaptativa
- `bordas/` - Detecção de bordas (Canny, Sobel, etc.)
- `morfologia/` - Operações morfológicas

### 04_caracteristicas/
**Extração de Características**
- `descritores/` - HOG, LBP, descritores de forma e textura
- `pontos_interesse/` - SIFT, SURF, ORB, FAST

### 05_reconhecimento/
**Reconhecimento de Padrões**
- Classificadores, clustering, reconhecimento de objetos

### 06_deep_learning/
**Visão Computacional Moderna**
- CNNs, transfer learning, detecção de objetos

## 🚀 Como Executar os Testes

### Teste Individual
```bash
# Navegar para o diretório do projeto
cd estudos_cv

# Ativar ambiente conda
conda activate estudos_cv

# Executar teste específico
python tests/01_fundamentos/conversoes_cinza/teste_completo_conversoes.py
```

### Teste por Tópico
```bash
# Executar todos os testes de um tópico
python -m tests.01_fundamentos  # (quando implementado)
```

## 📊 Convenções dos Testes

### Nomenclatura
- `teste_completo_*.py` - Teste abrangente de um tópico
- `benchmark_*.py` - Comparação de performance
- `validacao_*.py` - Validação contra implementações conhecidas
- `experimento_*.py` - Experimentos exploratórios

### Estrutura Padrão
Cada teste deve seguir esta estrutura:
```python
#!/usr/bin/env python3
"""Docstring explicando o propósito do teste."""

# Imports e setup de path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Função principal
def main():
    print("🧪 Nome do Teste")
    # ... código do teste ...

if __name__ == "__main__":
    main()
```

### Utilização dos Utils
- **`utils.visualization`** - Para plots padronizados e comparações
- **`utils.datasets`** - Para carregamento de imagens de teste
- **`utils.metrics`** - Para métricas de avaliação (quando implementado)

## 📈 Resultados

Os resultados dos testes são salvos em:
- `assets/results/` - Imagens e visualizações geradas
- Cada teste documenta seus próprios resultados

## 🎯 Objetivos Pedagógicos

1. **Validação** - Confirmar que implementações estão corretas
2. **Comparação** - Comparar diferentes abordagens
3. **Experimentação** - Explorar comportamentos com diferentes parâmetros
4. **Documentação** - Criar exemplos de uso da biblioteca

---
*Esta estrutura cresce conforme novos tópicos são implementados na ementa.*