import matplotlib.pyplot as plt
import pandas as pd


# Carrega o dataset Bank Marketing
df = pd.read_csv("data/bank-full.csv", sep=";")


# Informações gerais
print("=== DIMENSÕES DA BASE ===")
print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")


print("\n=== COLUNAS ===")
print(df.columns.tolist())


# Distribuição da variável alvo
print("\n=== DISTRIBUIÇÃO DA CLASSE (y) ===")
print(df["y"].value_counts())


print("\n=== DISTRIBUIÇÃO PERCENTUAL DA CLASSE ===")
print(df["y"].value_counts(normalize=True) * 100)


# Característica categórica
print("\n=== ESTADO CIVIL (marital) ===")
print(df["marital"].value_counts())


# Características contínuas
print("\n=== IDADE (age) ===")
print(df["age"].describe())


print("\n=== DURAÇÃO DA CHAMADA (duration) ===")
print(df["duration"].describe())


# Visualização das características escolhidas
print("\n=== DADOS SELECIONADOS ===")
print(df[["age", "duration", "marital", "y"]].head(10))

# ============================================================
# ANÁLISE DAS CARACTERÍSTICAS SEPARADAS POR CLASSE
# ============================================================

print("\n=== IDADE POR CLASSE ===")
print(df.groupby("y")["age"].describe())

print("\n=== DURAÇÃO POR CLASSE ===")
print(df.groupby("y")["duration"].describe())

print("\n=== ESTADO CIVIL POR CLASSE ===")
print(pd.crosstab(df["marital"], df["y"]))


# ------------------------------------------------------------
# Gráfico da idade
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df[df["y"] == "no"]["age"],
    bins=30,
    alpha=0.6,
    density=True,
    label="Não aderiu"
)

plt.hist(
    df[df["y"] == "yes"]["age"],
    bins=30,
    alpha=0.6,
    density=True,
    label="Aderiu"
)

plt.xlabel("Idade")
plt.ylabel("Densidade")
plt.title("Distribuição da idade por classe")
plt.legend()
plt.tight_layout()

plt.savefig("resultados/idade_por_classe.png")
plt.show()


# ------------------------------------------------------------
# Gráfico da duração
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df[df["y"] == "no"]["duration"],
    bins=50,
    alpha=0.6,
    density=True,
    label="Não aderiu"
)

plt.hist(
    df[df["y"] == "yes"]["duration"],
    bins=50,
    alpha=0.6,
    density=True,
    label="Aderiu"
)

plt.xlabel("Duração da chamada (segundos)")
plt.ylabel("Densidade")
plt.title("Distribuição da duração da chamada por classe")
plt.legend()
plt.tight_layout()

plt.savefig("resultados/duracao_por_classe.png")
plt.show()


# ------------------------------------------------------------
# Gráfico do estado civil
# ------------------------------------------------------------

tabela_marital = pd.crosstab(
    df["marital"],
    df["y"],
    normalize="columns"
)

tabela_marital.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.xlabel("Estado civil")
plt.ylabel("Proporção dentro da classe")
plt.title("Distribuição do estado civil por classe")
plt.legend(["Não aderiu", "Aderiu"])
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("resultados/estado_civil_por_classe.png")
plt.show()