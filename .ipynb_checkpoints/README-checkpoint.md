# Estudos em Visão Computacional - Implementação Manual

Este repositório documenta minha jornada de aprendizado em Visão Computacional, com foco na implementação manual de algoritmos clássicos e modernos a partir do zero, utilizando Python.

O objetivo principal é construir um entendimento profundo dos conceitos fundamentais, evitando o uso de funções "caixa-preta" de bibliotecas como OpenCV para as operações centrais de CV.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

- **`cv_lib/`**: Nossa biblioteca de Visão Computacional personalizada. Todas as funções implementadas (conversão de espaço de cor, filtros, detectores de borda, etc.) residem aqui.
- **`notebooks/`**: Contém os notebooks Jupyter que servem como um diário de laboratório. Cada notebook explora um tópico da ementa, combinando explicações teóricas, experimentação de código e visualização de resultados.
- **`assets/`**: Armazena arquivos de suporte, como imagens de teste.
- **`README.md`**: Este arquivo, com a descrição geral do projeto.

## Como Começar

Para configurar o ambiente e executar os experimentos, siga os passos abaixo.

1.  **Pré-requisitos**: É necessário ter o [Anaconda](https://www.anaconda.com/products/distribution) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instalado.

2.  **Criar o Ambiente Conda**:
    ```bash
    conda create --name estudos_cv python=3.9
    ```

3.  **Ativar o Ambiente**:
    ```bash
    conda activate estudos_cv
    ```

4.  **Instalar as Dependências**:
    ```bash
    pip install numpy matplotlib scikit-image jupyterlab
    ```

5.  **Iniciar o JupyterLab**:
    Navegue até a pasta raiz do projeto (`estudos_cv/`) e execute:
    ```bash
    jupyter lab
    ```
    Isso abrirá a interface do Jupyter no seu navegador, pronta para explorar os notebooks.