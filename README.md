# Deep Learning: Programming Assignment 1 — Segmentação de instâncias com as arquiteturas da aula

O objetivo desse repositório é atender ao enunciado do Programming Assignment 1 da disciplina Aprendizado Profundo da FGV EMAp. 

## Integrantes do Grupo

Henrique Gabriel Gasparelo \
José Thevez Gomes Guedes

## Estrutura do Repositório

A estrutura do repositório segue abaixo:

```text
.
├── checkpoints/
│   └── checkpoint.pt    # Pesos do modelo
├── data/
│   └── stage1_train/    # Dados reais - DSB2018
├── docs/
│   ├── DL_20242_07_SemanticSegmentation.pdf    # Slides da aula
│   └── PA1.pdf    # Enunciado do projeto
├── src/
│   ├── __init__.py 
│   ├── data.py    # Carregamento, separação dos dados e alvos da Trilha A
│   ├── failures.py    # Mineração de falhas e análise de campo receptivo
│   ├── inference.py    # Funções para o notebook de inferência
│   ├── losses.py    # Implementação de funções de perda
│   ├── metrics.py    # IoU, Dice, mAP@[.50:.95] e Algoritmo Húngaro
│   ├── models.py    # Modelos (ResUnet e SegNet)
│   ├── mosaic.py    # Tiling com overlap e fusão via DSU
│   ├── pipeline.ipynb    # Notebook do pipeline de ponta a ponta
│   ├── postprocessing.py    # Watershed e pós-processamento de instâncias
│   ├── stress.py    # Testes de estresse e perturbações fotométricas
│   ├── training.py    # Treinamento, validação e rotinas multi-seed
│   ├── utils.py    # Seleção de acelerador (MPS/CUDA/CPU), utilitários e criação de classes 
│   └── visualization.py    # Plots de gráficos e previsões
├── example.py    # Exemplo de treino e avaliação do modelo
├── inferencia.ipynb    # Notebook para testar o modelo em uma nova imagem
├── .gitignore
├── AI_LOG.md    # Descrição do uso de IA
└── README.md
```

O arquivo principal é `pipeline.ipynb`, onde está a resolução de todos os itens do enunciado. Os demais arquivos da pasta `src`são módulos com as funções utilizadas no `pipeline.ipynb`.

O diretório `data` contém os dados utilizados no treinamento, validação e teste do modelo. Os dados não estão incluídos no repositório, entretanto, podem ser encontrados [aqui](https://bbbc.broadinstitute.org/BBBC038).

O diretório `checkpoints` possui os pesos treinados do modelo. Para acessar o modelo com esses pesos utilize a função `carregar_checkpoint` do módulo `utils` usando o caminho do checkpoint e o dispositivo para alocar os tensores (cpu, cuda, mps, etc). 

O diretório `docs` possui o enunciado do projeto e os slides usados na aula. 

O arquivo `inferencia.ipynb` carrega o modelo e prevê as instâncias de uma imagem dada. Para testá-lo, basta substituir o caminho da imagem na variável `caminho_imagem` e rodar as duas células do notebook em ordem. O notebook mostrará as instâncias previstas e o número de instâncias detectadas.

## Treinamento e Avaliação do modelo

Caso o leitor queira treinar e avaliar o modelo, foram criadas classes e métodos para carregar dados, treinar o modelo e avaliar o modelo. 

No arquivo `example.py` está um exemplo de uso. 

A classe `Dataset` do módulo `utils` carrega e divide os dados. Essa classe recebe o caminho para os dados (que por padrão é "data/stage1_train") e possui os métodos: `get_data` que carrega os dados e `split_data` que divide os dados em treino, validação e teste, além de preparar as classes para o modelo da Trilha A.

A classe `Model` do módulo `utils` armazena, treina e avalia o modelo. Essa classe recebe o disposito para armazenar os tensores (cpu, cuda, mps, etc), apesar de inferir qual dispositivo está disponível quando nenhum argumento é passado. Além disso, a classe possui o método `train`, que recebe o dataset e treina o modelo, e o método `evaluate` que recebe o dataset e retorna as métricas utilizadas para avaliar o modelo.