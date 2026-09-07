# Classificador Bayesiano - Bank Marketing

Projeto desenvolvido para a disciplina de **Inteligência Artificial**, com o objetivo de implementar e analisar um classificador **Naive Bayes** utilizando o dataset Bank Marketing.

## Dataset

Foi utilizado o **Bank Marketing Dataset**, disponibilizado pelo UCI Machine Learning Repository.

O objetivo é prever se um cliente aderiu (`yes`) ou não (`no`) a um depósito a prazo após uma campanha de marketing bancário.

- Registros: 45.211
- Classe `0`: não aderiu
- Classe `1`: aderiu
- Distribuição: 88,3% classe 0 e 11,7% classe 1

Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Características utilizadas

Foram selecionadas exatamente três características:

| Característica | Tipo | Distribuição |
|---|---|---|
| `age` | Contínua | Normal |
| `duration` | Contínua | Gamma |
| `marital` | Categórica | Categórica |

A idade foi aproximada por uma distribuição Normal. A duração apresenta valores não negativos e assimetria à direita, sendo modelada por uma distribuição Gamma. O estado civil é representado por probabilidades categóricas.

Como existem valores de `duration = 0`, foi utilizada a transformação `duration + 1` para permitir a modelagem pela distribuição Gamma.

## Metodologia

O dataset foi dividido de forma estratificada em:

- 80% para treinamento;
- 20% para teste;
- `random_state = 42`.

Todos os priors, parâmetros das distribuições e probabilidades categóricas foram estimados **somente com os dados de treinamento**.

O projeto realiza:

- análise exploratória das características por classe;
- estimação das distribuições probabilísticas;
- cálculo de likelihood;
- cálculo da razão de verossimilhança;
- aplicação do Teorema de Bayes para obtenção das posteriores;
- determinação das regras e fronteiras de decisão;
- implementação manual do Naive Bayes;
- avaliação no conjunto de teste.

## Análise univariada

As probabilidades a priori estimadas no treino foram:

```text
P(Y=0) = 0.883018
P(Y=1) = 0.116982
```

As fronteiras de decisão encontradas para as características contínuas foram aproximadamente:

```text
age      = 72,91 anos
duration = 785 segundos
```

Entre as três características, `duration` apresentou a separação mais evidente entre as classes. `age` apresenta grande sobreposição entre as distribuições, enquanto `marital` fornece alguma evidência sobre a classe, mas não é suficiente isoladamente para superar o forte prior da classe 0.

Os gráficos das distribuições e das fronteiras de decisão são gerados na pasta `resultados/`.

## Naive Bayes

O classificador combina as três características:

```text
age + duration + marital
```

O Naive Bayes assume **independência condicional entre as características dada a classe**.

A implementação foi realizada manualmente utilizando log-probabilidades. Dessa forma, em vez de multiplicar diversas probabilidades pequenas diretamente, são somados seus logaritmos, reduzindo problemas de precisão numérica.

A classe com maior log-score é escolhida pelo modelo.

## Resultados

O conjunto de teste possui 9.043 registros, sendo:

```text
Classe 0: 7985 (88,30%)
Classe 1: 1058 (11,70%)
```

A matriz de confusão obtida foi:

```text
[[7797  188]
 [ 826  232]]
```

| Métrica | Resultado |
|---|---:|
| Acurácia | 88,79% |
| Precisão | 55,24% |
| Recall | 21,93% |
| F1-score | 0,3139 |
| Baseline | 88,30% |

O modelo identificou corretamente **232 dos 1.058 positivos reais**.

Apesar da acurácia de 88,79%, o baseline que sempre prevê a classe 0 já alcança 88,30%. Por isso, a acurácia deve ser analisada em conjunto com precisão, recall, F1-score e matriz de confusão.

O recall de 21,93% mostra que o modelo ainda possui dificuldade para identificar a classe minoritária.

## Estrutura

```text
├── data/
│   └── bank-full.csv
│
├── resultados/
│   ├── duracao_por_classe.png
│   ├── estado_civil_por_classe.png
│   ├── fronteira_duracao.png
│   ├── fronteira_idade.png
│   ├── idade_por_classe.png
│   └── matriz_confusao.png
│
├── src/
│   ├── modelo/
│   │   ├── __init__.py
│   │   ├── bayes.py
│   │   └── dados.py
│   ├── 01_exploracao.py
│   ├── 02_analise_univariada.py
│   ├── 03_naive_bayes.py
│   └── 04_avaliacao.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Como executar

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Execute os scripts na ordem:

```powershell
python src/01_exploracao.py
python src/02_analise_univariada.py
python src/03_naive_bayes.py
python src/04_avaliacao.py
```

## Tecnologias

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Scikit-learn

## Limitações

O dataset apresenta forte desbalanceamento entre as classes, fazendo com que o prior favoreça significativamente a classe 0.

Além disso, o Naive Bayes assume independência condicional entre `age`, `duration` e `marital`, o que é uma simplificação das relações existentes nos dados reais.

As distribuições Normal e Gamma também são aproximações probabilísticas das características observadas.

A variável `duration` só é conhecida após a realização da ligação. Portanto, seu uso é adequado para este estudo acadêmico, mas seria uma limitação em um sistema utilizado para decidir antecipadamente quais clientes deveriam ser contatados.

## Autores

- Vinicius Leite Costa
- Gison Vilaça

## Apresentação

Vídeo de apresentação: