# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Portuguese academic computer vision study project** with a strong pedagogical focus. The primary goal is **learning through manual implementation** of classical and modern CV algorithms from scratch, building deep conceptual understanding by avoiding "black-box" functions from libraries like OpenCV.

### 🎯 Pedagogical Mission
This project prioritizes **education over efficiency**. Every implementation should:
- **Explain concepts first**: All notebooks must include detailed theoretical explanations in Portuguese using Markdown cells BEFORE any code implementation
- **Document thoroughly**: Code should be extensively documented with clear comments explaining the "why" and "how" 
- **Visualize results**: Include matplotlib visualizations to demonstrate algorithm behavior
- **Build incrementally**: Start with simple concepts and build complexity gradually
- **Compare approaches**: Show differences between manual implementations and library functions when applicable

## Environment Setup

This project uses Conda for environment management:

```bash
# Create environment
conda create --name estudos_cv python=3.9

# Activate environment  
conda activate estudos_cv

# Install dependencies
pip install numpy matplotlib scikit-image jupyterlab

# Start JupyterLab
jupyter lab
```

## Project Architecture

### Directory Structure
- **`cv_lib/`**: Custom computer vision library containing all manual implementations
  - `processamento.py`: Core image processing functions (RGB to grayscale conversion, etc.)
  - `__init__.py`: Package initialization
- **`notebooks/`**: Jupyter notebooks serving as laboratory diary with theoretical explanations and experiments
- **`assets/`**: Test images and support files (currently empty)

### Code Architecture

The project follows a modular approach where:

1. **Custom Library (`cv_lib/`)**: Contains manual implementations of CV algorithms
   - Functions use NumPy arrays for image representation
   - Images are expected as `(height, width)` for grayscale or `(height, width, 3)` for RGB
   - All implementations avoid using OpenCV for core operations
   - Uses `np.uint8` dtype for pixel values (0-255 range)

2. **Notebooks**: Import from `cv_lib` and demonstrate usage
   - Include path setup to import custom library: 
   ```python
   import sys, os
   project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
   sys.path.insert(0, project_root)
   from cv_lib import processamento
   ```

### Current Implementation Status

- **Basic image processing**: RGB to grayscale conversion using luminance formula (0.299*R + 0.587*G + 0.114*B)
- **Image representation**: Uses NumPy arrays with proper dtype handling
- **Notebook structure**: Theory in Markdown → Implementation → Experimentation pattern established
- **Visualization**: Uses matplotlib for displaying results with proper comparisons

## Development Guidelines - Pedagogical Focus

### 🔬 Notebook Structure (MANDATORY)
Every notebook must follow this pedagogical structure:
1. **Theoretical Introduction (Markdown)**: Explain concepts in Portuguese before any code
   - What is the algorithm/technique?
   - Why is it important?
   - Mathematical foundations when applicable
   - Real-world applications
2. **Manual Implementation**: Step-by-step implementation with extensive comments
3. **Experimentation**: Test with different parameters and visualize results
4. **Comparison**: When possible, compare with library implementations to validate correctness

### 💻 Code Implementation Standards
- **Portuguese comments**: All code comments and docstrings in Portuguese
- **Educational code**: Prefer clarity over performance (use explicit loops instead of vectorized operations when it aids understanding)
- **Incremental building**: Break complex algorithms into smaller, understandable functions
- **Visual feedback**: Always include matplotlib visualizations showing intermediate and final results
- **Error handling**: Include proper error handling with educational messages

### 📚 Documentation Requirements
- **Rich docstrings**: Explain parameters, return values, and algorithm steps
- **Conceptual explanations**: Include mathematical formulas and theoretical background
- **Usage examples**: Provide clear examples of how to use each function
- **Visual documentation**: Include diagrams or visual explanations when helpful

### 🧪 Testing and Validation
- Test implementations in Jupyter notebooks with real images
- Compare results with known-good implementations when possible
- Include edge cases and error conditions in testing
- Visualize intermediate steps to aid understanding

## Technical Architecture

- **Manual implementations only**: No OpenCV for core CV operations (only numpy, matplotlib, scipy when absolutely necessary)
- **Object-oriented design**: Use classes for complex algorithms, functions for simple operations  
- **NumPy-centric**: Use NumPy arrays for all image operations (uint8 dtype, proper shapes)
- **Modular structure**: Keep implementations in `cv_lib/`, experiments in `notebooks/`
- **Portuguese-first**: All documentation, comments, and explanations in Portuguese

## 📚 CONSULTA OBRIGATÓRIA À BASE CV

**REGRA FUNDAMENTAL:** Antes de implementar QUALQUER conceito novo de visão computacional:

### 1. 🔍 PESQUISA OBRIGATÓRIA
```
SEMPRE use as seguintes tools MCP para pesquisar teoria ANTES de implementar:

1. mcp__cv-books__cv_base - BASE PRINCIPAL (OBRIGATÓRIA)
   - Consulta aos livros acadêmicos de referência
   - Conceitos fundamentais
   - Fórmulas matemáticas exatas
   - Aplicações práticas
   - Valores de parâmetros típicos
   - Equações numeradas e páginas específicas

2. mcp__cv-books__wikipedia-api - COMPLEMENTAR (Opcional)
   - Contexto geral e introdutório
   - Conceitos básicos
   - Histórico e aplicações
   - Recursos visuais e diagramas

FLUXO OBRIGATÓRIO:
1º - Pesquisar em mcp__cv-books__cv_base (livros acadêmicos)
2º - Se necessário, complementar com wikipedia-api
3º - Documentar TODAS as referências encontradas
```

### 2. 📖 DOCUMENTAÇÃO ACADÊMICA OBRIGATÓRIA
Todo código deve incluir referências específicas:
```python
"""
Função: [Nome da Operação]

📚 **Referência:** [Autor], [Livro], [Edição], [Capítulo], [Equação], [Página]
                  [Citação direta relevante]

🧮 **Fórmula Matemática:** [Equação exata da literatura]
    onde [definição de cada variável]

🎯 **Aplicações Citadas:** [Usos mencionados na literatura]
🔢 **Parâmetros Típicos:** [Valores sugeridos pelos autores]
⚠️ **Implementação:** [Considerações técnicas importantes]
"""
```

### 3. 📚 FONTES ACADÊMICAS PRIORITÁRIAS
1. **Gonzalez & Woods** - Digital Image Processing 4e
2. **Szeliski** - Computer Vision: Algorithms and Applications 2e
3. **Gonzalez, Woods & Eddins** - Digital Image Processing Using MATLAB 2e

### 4. ✅ VALIDAÇÃO OBRIGATÓRIA
- Verificar fórmulas contra literatura acadêmica
- Confirmar ranges de parâmetros citados
- Incluir exemplos de aplicação mencionados
- Referenciar números de página e equações específicas

### 5. 🚫 NUNCA IMPLEMENTAR SEM:
- Pesquisar primeiro na base CV
- Documentar referências específicas
- Validar contra fontes acadêmicas
- Incluir citações diretas relevantes

**Objetivo:** Garantir que toda implementação seja 100% fundamentada academicamente com referências rastreáveis.

## Curriculum Coverage

The project systematically covers CV fundamentals through advanced topics:
1. **Image processing fundamentals**: Color spaces, point operations, histograms, spatial/frequency filtering
2. **Segmentation and detection**: Thresholding, edge detection, Hough transforms, morphological operations
3. **Feature extraction**: HOG, LBP, SIFT, SURF, ORB implementations from scratch
4. **Pattern recognition**: Manual implementation of classifiers and clustering algorithms
5. **Modern CV**: CNN architectures, object detection, segmentation (understanding-focused implementations)
6. **Advanced topics**: Self-supervised learning concepts, transformer architectures, 3D vision fundamentals