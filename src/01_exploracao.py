import os
import pandas as pd
import matplotlib.pyplot as plt

ARQUIVO = "data/bank-full.csv"

os.makedirs("resultados", exist_ok=True)

df = pd.read_csv(ARQUIVO, sep=";")

print("=== DATASET ===")
print(f"Registros: {len(df)}")
print(f"Colunas: {len(df.columns)}")

print("\n=== CLASSES ===")
print(df["y"].value_counts())
print(df["y"].value_counts(normalize=True) * 100)

print("\n=== AGE POR CLASSE ===")
print(
    df.groupby("y")["age"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print("\n=== DURATION POR CLASSE ===")
print(
    df.groupby("y")["duration"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print("\n=== MARITAL POR CLASSE ===")
print(pd.crosstab(df["marital"], df["y"]))


def salvar_histograma(coluna, arquivo, titulo):
    for classe in ["no", "yes"]:
        dados = df[df["y"] == classe][coluna]

        plt.hist(
            dados,
            bins=40,
            density=True,
            alpha=0.5,
            label=classe
        )

    plt.title(titulo)
    plt.xlabel(coluna)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/{arquivo}")
    plt.close()


salvar_histograma(
    "age",
    "idade_por_classe.png",
    "Idade por classe"
)

salvar_histograma(
    "duration",
    "duracao_por_classe.png",
    "Duração por classe"
)

pd.crosstab(
    df["marital"],
    df["y"],
    normalize="index"
).plot(kind="bar")

plt.title("Estado civil por classe")
plt.ylabel("Proporção")
plt.tight_layout()
plt.savefig("resultados/estado_civil_por_classe.png")
plt.close()

print("\nGráficos salvos em resultados/")