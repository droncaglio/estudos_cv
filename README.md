# 🔍 Estudos em Visão Computacional - Implementação Manual

## 📖 Sobre o Projeto

Este repositório documenta uma jornada de aprendizado **acadêmico** em Visão Computacional, com ênfase na **implementação manual** de algoritmos clássicos e modernos a partir do zero, utilizando Python e princípios de orientação a objetos.

### 🎯 Objetivo Pedagógico

O foco principal é **construir entendimento profundo** dos conceitos fundamentais através de:
- **Implementação from-scratch**: Evitar funções "caixa-preta" de bibliotecas como OpenCV para operações centrais
- **Documentação rica**: Código bem documentado com explicações detalhadas dos conceitos
- **Explicações teóricas**: Notebooks Jupyter com teoria em Markdown antes de cada implementação
- **Visualizações**: Demonstrações práticas dos resultados de cada algoritmo
- **Arquitetura orientada a objetos**: Código estruturado seguindo boas práticas de desenvolvimento

## 📚 Ementa Acadêmica

### 1. 🔤 Fundamentos de Processamento de Imagens
- **Conceitos básicos**: Imagem digital, aquisição, amostragem e quantização
- **Espaços de cor**: RGB, HSV, YCbCr, Lab, escala de cinza
- **Operações pontuais**: Brilho, contraste, correção gama, normalização
- **Histogramas**: Equalização, especificação, equalização adaptativa (CLAHE)
- **Filtragem espacial**: Convolução, suavização (média, gaussiana), realce de bordas (Sobel, Prewitt, Laplaciano)
- **Filtragem na frequência**: Transformada de Fourier, filtros passa-baixa e passa-alta

### 2. 🎯 Segmentação e Detecção de Estruturas
- **Limiarização**: Global (Otsu) e adaptativa
- **Detecção de bordas**: Canny, Marr-Hildreth
- **Transformadas**: Hough para linhas e círculos
- **Segmentação por regiões**: Crescimento de regiões, watershed
- **Morfologia matemática**: Erosão, dilatação, abertura, fechamento, gradiente morfológico

### 3. 🔍 Representação e Descrição
- **Extração de características**: Cor, forma e textura (momentos, contornos, descritores)
- **Descritores avançados**: HOG, LBP
- **Pontos de interesse**: SIFT, SURF, ORB, FAST
- **Correspondência**: Feature matching

### 4. 🧠 Reconhecimento de Padrões
- **Classificação supervisionada**: k-NN, SVM, Random Forest
- **Agrupamento**: k-means, Mean Shift, DBSCAN
- **Reconhecimento**: Objetos e faces
- **Redução de dimensionalidade**: PCA

### 5. 🤖 Visão Computacional Moderna
- **CNNs**: LeNet, AlexNet, VGG, ResNet
- **Regularização**: Dropout, batch normalization
- **Transfer learning**: Fine-tuning de modelos
- **Detecção de objetos**: R-CNN, YOLO, SSD, RetinaNet
- **Segmentação**: FCN, U-Net, Mask R-CNN, DeepLab

### 6. 🚀 Tópicos Avançados
- **Aprendizado contrastivo**: SimCLR, MoCo
- **Transformers**: Vision Transformers (ViT), DETR
- **Reconhecimento facial**: FaceNet, ArcFace
- **Visão 3D**: SLAM, nuvens de pontos
- **Análise de movimento**: Fluxo óptico, tracking

### 7. 🛠️ Projetos Práticos
- Reconhecimento de faces e emoções
- Detecção de pedestres e veículos
- Segmentação de folhas

## 📖 Bibliografia

**Referências principais:**
- **Gonzalez, R. C.; Woods, R. E.** – *Digital Image Processing*. Pearson, 4ª ed. (2018)
- **Szeliski, R.** – *Computer Vision: Algorithms and Applications*. Springer, 2ª ed. (2022) [Acesso aberto]
- **Goodfellow, I.; Bengio, Y.; Courville, A.** – *Deep Learning*. MIT Press (2016) [Online gratuito]
- **Bishop, C. M.** – *Pattern Recognition and Machine Learning*. Springer (2006)

## 🏗️ Estrutura do Projeto

```
estudos_cv/
├── cv_lib/                    # 📚 Nossa biblioteca CV personalizada
│   ├── __init__.py           # Inicialização do pacote
│   └── processamento.py      # Implementações manuais dos algoritmos
├── notebooks/                # 📔 Diário de laboratório interativo
│   └── 01-Fundamentos_...   # Notebooks com teoria + prática
├── assets/                   # 🖼️ Imagens de teste e recursos
└── README.md                 # 📋 Este arquivo
```

### 📂 Descrição dos Diretórios

- **`cv_lib/`**: Nossa biblioteca de Visão Computacional **implementada do zero**
  - Todas as funções (conversão de espaço de cor, filtros, detectores, etc.)
  - Código bem documentado seguindo princípios OOP
  - Implementações manuais evitando "caixas-pretas"

- **`notebooks/`**: **Diário de laboratório acadêmico**
  - Cada notebook = um tópico da ementa
  - Estrutura pedagógica: **Teoria (Markdown) → Implementação → Experimentação**
  - Visualizações e comparações de resultados
  - Explicações conceituais detalhadas

- **`assets/`**: Recursos de apoio (imagens de teste, datasets, etc.)

## 🚀 Como Começar

### Pré-requisitos
- [Anaconda](https://www.anaconda.com/products/distribution) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Setup do Ambiente

```bash
# 1. Criar ambiente isolado
conda create --name estudos_cv python=3.9

# 2. Ativar ambiente
conda activate estudos_cv

# 3. Instalar dependências essenciais
pip install numpy matplotlib scikit-image jupyterlab

# 4. Iniciar JupyterLab (na raiz do projeto)
cd estudos_cv/
jupyter lab
```

### 📖 Como Estudar

1. **Explore os notebooks em ordem sequencial**
2. **Leia as explicações teóricas** (células Markdown) antes do código
3. **Execute e experimente** com o código implementado
4. **Compare resultados** entre implementação manual vs. bibliotecas
5. **Modifique parâmetros** para entender o comportamento dos algoritmos

## ✅ Progresso Detalhado

### 🏗️ Infraestrutura do Projeto
- [x] **Estrutura de diretórios** - Organização pedagógica completa
- [x] **Ambiente conda** - Python 3.9 + dependências
- [x] **Biblioteca cv_lib** - Pacote personalizado com importações funcionando
- [x] **Sistema de testes** - Estrutura organizada por tópicos em `/tests/`
- [x] **Utilitários** - Visualização e datasets centralizados em `/utils/`
- [x] **Documentação** - README, CLAUDE.md, estrutura navegável

### 1️⃣ Fundamentos de Processamento de Imagens

#### 1.1 Conceitos Básicos ✅
- [x] **Imagem digital** - Conceitos de amostragem e quantização
- [x] **Representação NumPy** - Arrays, dtypes, shapes

#### 1.2 Espaços de Cor ✅ 
- [x] **Modelo RGB** - Compreensão e manipulação
- [x] **Conversões para escala de cinza** - 7 métodos implementados:
  - [x] `luminancia/bt601` - Padrão clássico (ITU-R BT.601)  
  - [x] `bt709` - Padrão HDTV moderno
  - [x] `media` - Média aritmética simples
  - [x] `desaturacao` - (max + min) / 2
  - [x] `canal_r/g/b` - Canais individuais com aplicações específicas
- [x] **Testes implementados** - Comparação visual e quantitativa
- [x] **Documentação técnica** - Aplicações práticas de cada método

#### 1.3 Operações Pontuais 🔄 **PRÓXIMO**
- [ ] **Brilho e contraste** - Transformações lineares
- [ ] **Correção gama** - Ajuste não-linear
- [ ] **Normalização** - Padronização de valores
- [ ] **Operações aritméticas** - Soma, multiplicação entre imagens

#### 1.4 Histogramas
- [ ] **Cálculo de histogramas** - Distribuição de intensidades
- [ ] **Equalização** - Melhoria de contraste
- [ ] **Equalização adaptativa (CLAHE)** - Processamento local
- [ ] **Especificação de histograma** - Matching

#### 1.5 Filtragem Espacial
- [ ] **Convolução 2D** - Implementação manual
- [ ] **Filtros de suavização** - Média, Gaussiano
- [ ] **Realce de bordas** - Sobel, Prewitt, Laplaciano
- [ ] **Filtros customizados** - Kernels personalizados

### 2️⃣ Filtragem na Frequência
- [ ] **Transformada de Fourier** - FFT 2D
- [ ] **Filtros passa-baixa/alta** - No domínio da frequência
- [ ] **Filtragem butterworth** - Filtros ideais e práticos

### 3️⃣ Segmentação e Detecção
- [ ] **Limiarização global** - Otsu
- [ ] **Limiarização adaptativa** - Métodos locais
- [ ] **Detecção de bordas** - Canny, Marr-Hildreth
- [ ] **Transformadas** - Hough para linhas e círculos
- [ ] **Morfologia matemática** - Erosão, dilatação, abertura, fechamento

### 4️⃣ Características e Descritores
- [ ] **HOG** - Histogram of Oriented Gradients
- [ ] **LBP** - Local Binary Patterns
- [ ] **SIFT/SURF/ORB** - Pontos de interesse
- [ ] **Momentos** - Descritores de forma

### 5️⃣ Reconhecimento de Padrões
- [ ] **k-NN** - Classificação por vizinhos
- [ ] **SVM** - Support Vector Machines
- [ ] **k-means** - Agrupamento
- [ ] **PCA** - Redução de dimensionalidade

### 6️⃣ Deep Learning
- [ ] **CNNs básicas** - LeNet, arquiteturas fundamentais
- [ ] **Transfer Learning** - Modelos pré-treinados
- [ ] **Detecção de objetos** - YOLO, R-CNN
- [ ] **Segmentação** - U-Net, masks

---
**📊 Status Geral: 15/60+ funcionalidades implementadas (~25% da ementa)**