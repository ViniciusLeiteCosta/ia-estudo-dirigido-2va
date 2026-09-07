# Classificador Bayesiano - Bank Marketing

Projeto desenvolvido para o **Estudo Dirigido** da disciplina de **Inteligência Artificial** do curso de **Bacharelado em Ciência da Computação** da **Universidade Federal do Agreste de Pernambuco (UFAPE)**.

O projeto tem como objetivo aplicar conceitos de **classificação Bayesiana** e implementar manualmente um classificador **Naive Bayes**, utilizando características contínuas e categóricas.

---

## Informações

- **Disciplina:** Inteligência Artificial
- **Atividade:** Estudo Dirigido
- **Período:** 2026.1
- **Instituição:** Universidade Federal do Agreste de Pernambuco (UFAPE)

### Docente

- [Luis Filipe Alves Pereira](https://github.com/luisfilipeap)

### Discentes

- [Gison Vilaça](https://github.com/gison-vilaca)
- [Vinicius Leite](https://github.com/ViniciusLeiteCosta)

---

# Sobre o Projeto

O projeto consiste no desenvolvimento e análise de um **classificador Bayesiano** para prever se um cliente de uma instituição bancária aderiu ou não a um depósito a prazo após uma campanha de marketing.

Foi utilizado o **Bank Marketing Dataset**, disponibilizado pelo **UCI Machine Learning Repository**.

A variável alvo é `y`, representada no projeto por duas classes:

- **Classe 0:** o cliente não aderiu ao depósito (`no`);
- **Classe 1:** o cliente aderiu ao depósito (`yes`).

O dataset possui **45.211 registros** e apresenta um forte desbalanceamento entre as classes:

- aproximadamente **88,3%** pertencem à classe 0;
- aproximadamente **11,7%** pertencem à classe 1.

Dataset utilizado:

https://archive.ics.uci.edu/dataset/222/bank+marketing

---

# Características Utilizadas

Foram selecionadas exatamente três características para a construção do classificador:

| Característica | Tipo | Modelo probabilístico |
|----------------|------|-----------------------|
| `age` | Contínua | Distribuição Normal |
| `duration` | Contínua | Distribuição Gamma |
| `marital` | Categórica | Probabilidades categóricas |

## Age

Representa a idade do cliente.

Foi modelada utilizando uma **distribuição Normal** para cada classe. A análise mostrou uma grande sobreposição entre as distribuições das duas classes, indicando baixo poder de discriminação quando utilizada isoladamente.

## Duration

Representa a duração da última ligação realizada com o cliente, em segundos.

Foi utilizada uma **distribuição Gamma**, pois a variável apresenta valores não negativos e uma distribuição assimétrica à direita.

Como existem observações com `duration = 0`, foi utilizada a transformação:

```text
duration + 1
```

para permitir o ajuste da distribuição Gamma utilizada no projeto.

Entre as três características selecionadas, `duration` apresentou o maior poder de discriminação entre as classes.

## Marital

Representa o estado civil do cliente e possui as categorias:

- `married`;
- `single`;
- `divorced`.

Por ser uma característica categórica, foram utilizadas diretamente as probabilidades condicionais de cada categoria em cada classe.

---

# Divisão dos Dados

O dataset foi dividido de forma estratificada em:

- **80% para treinamento**;
- **20% para teste**;
- semente aleatória (`random_state`) igual a **42**.

A divisão resultou em:

| Conjunto | Registros |
|----------|----------:|
| Treino | 36.168 |
| Teste | 9.043 |

Distribuição das classes no conjunto de treinamento:

| Classe | Registros | Proporção |
|--------|----------:|----------:|
| 0 | 31.937 | 88,30% |
| 1 | 4.231 | 11,70% |

Distribuição das classes no conjunto de teste:

| Classe | Registros | Proporção |
|--------|----------:|----------:|
| 0 | 7.985 | 88,30% |
| 1 | 1.058 | 11,70% |

Todos os **priors, parâmetros das distribuições e probabilidades categóricas** utilizados pelo classificador foram estimados exclusivamente com os dados de treinamento.

---

# Análise Bayesiana

Para cada característica foi realizada individualmente uma análise Bayesiana contendo:

- estimação das distribuições condicionais por classe;
- cálculo das verossimilhanças;
- cálculo da razão de verossimilhanças;
- estimação das probabilidades a priori;
- cálculo das probabilidades a posteriori;
- definição das regras de decisão;
- determinação das fronteiras de decisão das características contínuas.

As probabilidades a priori estimadas no conjunto de treinamento foram:

```text
P(Y=0) = 0.883018
P(Y=1) = 0.116982
```

As fronteiras de decisão encontradas para as características contínuas foram aproximadamente:

```text
age      = 72,91 anos
duration = 785 segundos
```

Os gráficos gerados permitem visualizar as distribuições condicionais, as regiões de decisão e suas respectivas fronteiras.

---

# Naive Bayes

Após a análise individual das características, foi implementado manualmente um classificador **Naive Bayes** utilizando simultaneamente:

```text
age + duration + marital
```

O modelo assume **independência condicional entre as características dada a classe**.

Dessa forma, a evidência fornecida pelas três características pode ser combinada para calcular um score para cada classe.

A implementação utiliza **log-probabilidades**, evitando a multiplicação direta de probabilidades muito pequenas e reduzindo problemas de precisão numérica.

A classe que apresenta o maior log-score é utilizada como previsão final.

---

# Funcionalidades

O projeto realiza:

- carregamento e preparação do Bank Marketing Dataset;
- transformação da variável alvo;
- divisão estratificada em treino e teste;
- análise exploratória das características;
- visualização das características por classe;
- estimação das distribuições probabilísticas;
- cálculo de likelihoods;
- cálculo da razão de verossimilhanças;
- cálculo das probabilidades a posteriori;
- determinação das fronteiras de decisão;
- geração dos gráficos das regiões de decisão;
- implementação manual do Naive Bayes;
- combinação de características contínuas e categóricas;
- previsão no conjunto de teste;
- geração da matriz de confusão;
- cálculo de acurácia, precisão, recall e F1-score.

---

# Resultados

A avaliação final foi realizada exclusivamente no conjunto de teste.

A matriz de confusão obtida foi:

```text
[[7797  188]
 [ 826  232]]
```

Portanto:

| Resultado | Quantidade |
|-----------|-----------:|
| Verdadeiros Negativos (TN) | 7.797 |
| Falsos Positivos (FP) | 188 |
| Falsos Negativos (FN) | 826 |
| Verdadeiros Positivos (TP) | 232 |

As métricas obtidas foram:

| Métrica | Resultado |
|---------|----------:|
| Acurácia | 88,79% |
| Precisão | 55,24% |
| Recall | 21,93% |
| F1-score | 0,3139 |
| Baseline | 88,30% |

Apesar da acurácia de **88,79%**, o baseline que sempre prevê a classe majoritária já alcança aproximadamente **88,30%**.

O resultado deve, portanto, ser analisado juntamente com as demais métricas.

O principal erro observado foi o número de **falsos negativos**. Dos **1.058 clientes que realmente aderiram**, apenas **232 foram identificados corretamente**, resultando em um recall de **21,93%**.

---

# Visualizações

Durante a execução são geradas visualizações na pasta `resultados/`:

- distribuição de idade por classe;
- distribuição da duração por classe;
- distribuição do estado civil por classe;
- distribuição e fronteira de decisão da idade;
- distribuição e fronteira de decisão da duração;
- matriz de confusão.

---

# Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Scikit-learn

---

# Estrutura do Projeto

```text
.
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
│   │
│   ├── 01_exploracao.py
│   ├── 02_analise_univariada.py
│   ├── 03_naive_bayes.py
│   └── 04_avaliacao.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

| Arquivo | Descrição |
|---------|-----------|
| `src/modelo/dados.py` | Carregamento, divisão dos dados e estimação dos parâmetros |
| `src/modelo/bayes.py` | Funções utilizadas pelo classificador Bayesiano |
| `src/01_exploracao.py` | Análise exploratória e visualizações iniciais |
| `src/02_analise_univariada.py` | Análise Bayesiana individual das três características |
| `src/03_naive_bayes.py` | Combinação das características e classificação Naive Bayes |
| `src/04_avaliacao.py` | Avaliação no conjunto de teste e matriz de confusão |

---

# Instalação

Clone o repositório:

```bash
git clone https://github.com/ViniciusLeiteCosta/ia-estudo-dirigido-2va.git
```

Entre na pasta do projeto:

```bash
cd ia-estudo-dirigido-2va
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual.

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

# Execução

Execute os scripts na seguinte ordem:

```bash
python src/01_exploracao.py
python src/02_analise_univariada.py
python src/03_naive_bayes.py
python src/04_avaliacao.py
```

Os gráficos e resultados visuais serão armazenados na pasta `resultados/`.

---

# Aspectos de Inteligência Artificial

Durante o desenvolvimento foram aplicados conceitos como:

- classificação supervisionada;
- classificação Bayesiana;
- distribuições condicionais;
- probabilidade a priori;
- verossimilhança;
- razão de verossimilhanças;
- Teorema de Bayes;
- probabilidade a posteriori;
- regras de decisão Bayesiana;
- fronteiras de decisão;
- independência condicional;
- Naive Bayes;
- classificação com características contínuas e categóricas;
- matriz de confusão;
- avaliação de classificadores.

---

# Limitações

O modelo possui algumas limitações importantes.

O dataset apresenta um forte **desbalanceamento entre as classes**, fazendo com que a probabilidade a priori favoreça significativamente a classe 0.

A distribuição Normal utilizada para `age` e a distribuição Gamma utilizada para `duration` são aproximações das distribuições reais observadas.

O Naive Bayes também assume independência condicional entre `age`, `duration` e `marital`, hipótese que pode não ser completamente verdadeira nos dados reais.

Além disso, apenas três características do dataset foram utilizadas no classificador.

A variável `duration` apresenta uma limitação adicional: sua duração completa somente é conhecida após a realização da ligação. Portanto, seu uso é adequado para este estudo acadêmico, mas seria inadequado em um sistema cujo objetivo fosse decidir antecipadamente quais clientes deveriam ser contatados.

---

# Apresentação

Vídeo de apresentação do projeto: https://drive.google.com/file/d/1d65De-7SDvxTzoGMh4WNCYwtnjG6pKri/view?usp=sharing



---

# Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos como atividade da disciplina de Inteligência Artificial da UFAPE.
