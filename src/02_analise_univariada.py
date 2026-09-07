import os
import numpy as np
import matplotlib.pyplot as plt

from modelo.dados import preparar_dados
from modelo.bayes import likelihood, posterior

df, _, _, _, _, p = preparar_dados()

os.makedirs("resultados", exist_ok=True)

print("=== ANÁLISE BAYESIANA UNIVARIADA ===")

print("\n=== PRIORS ===")
for c in (0, 1):
    print(f"P(Y={c}) = {p['prior'][c]:.6f}")


print("\n=== PARÂMETROS ESTIMADOS ===")

for c in (0, 1):
    media, desvio = p["age"][c]
    shape, loc, scale = p["duration"][c]

    print(f"\nClasse {c}:")
    print(f"age -> média={media:.4f}, desvio={desvio:.4f}")
    print(
        f"duration -> shape={shape:.6f}, "
        f"loc={loc:.6f}, scale={scale:.6f}"
    )

    print("marital:")
    for categoria, prob in p["marital"][c].items():
        print(f"  {categoria}: {prob:.6f}")


def analisar(feature, valores):
    print(f"\n=== {feature.upper()} ===")

    for valor in valores:
        l0 = likelihood(feature, valor, 0, p)
        l1 = likelihood(feature, valor, 1, p)
        post = posterior(feature, valor, p)

        print(f"\nValor: {valor}")
        print(f"Likelihood 0 = {l0:.8f}")
        print(f"Likelihood 1 = {l1:.8f}")
        print(f"LR (1/0) = {l1 / l0:.4f}")
        print(f"Posterior 0 = {post[0]:.2%}")
        print(f"Posterior 1 = {post[1]:.2%}")
        print(f"Classe = {np.argmax(post)}")


analisar("age", [30, 40, 60])
analisar("duration", [60, 300, 600])
analisar("marital", ["divorced", "married", "single"])


def encontrar_fronteira(feature, inicio, fim):
    valores = np.linspace(inicio, fim, 30000)

    diferenca = [
        p["prior"][1] * likelihood(feature, x, 1, p)
        - p["prior"][0] * likelihood(feature, x, 0, p)
        for x in valores
    ]

    indices = np.where(np.diff(np.sign(diferenca)))[0]

    return valores[indices]


idade = encontrar_fronteira(
    "age",
    df["age"].min(),
    df["age"].max()
)

duracao = encontrar_fronteira(
    "duration",
    df["duration"].min(),
    df["duration"].max()
)

print("\n=== FRONTEIRAS ===")
print(f"Idade: {idade[0]:.2f} anos")
print(f"Duração: {duracao[0]:.2f} segundos")

print("\nRegras:")
print(f"age <= {idade[0]:.2f} -> classe 0")
print(f"age > {idade[0]:.2f} -> classe 1")

print(f"duration <= {duracao[0]:.2f} -> classe 0")
print(f"duration > {duracao[0]:.2f} -> classe 1")

for estado in p["marital"][0]:
    classe = np.argmax(posterior("marital", estado, p))
    print(f"{estado} -> classe {classe}")


def grafico_fronteira(feature, inicio, fim, fronteira, arquivo, titulo):
    x = np.linspace(inicio, fim, 1000)

    y0 = [likelihood(feature, valor, 0, p) for valor in x]
    y1 = [likelihood(feature, valor, 1, p) for valor in x]

    plt.plot(x, y0, label="Classe 0")
    plt.plot(x, y1, label="Classe 1")

    plt.axvspan(
        inicio,
        fronteira,
        alpha=0.1,
        label="Decisão: classe 0"
    )

    plt.axvspan(
        fronteira,
        fim,
        alpha=0.1,
        label="Decisão: classe 1"
    )

    plt.axvline(
        fronteira,
        linestyle="--",
        label=f"Fronteira = {fronteira:.2f}"
    )

    plt.title(titulo)
    plt.xlabel(feature)
    plt.ylabel("Densidade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/{arquivo}")
    plt.close()


grafico_fronteira(
    "age",
    df["age"].min(),
    df["age"].max(),
    idade[0],
    "fronteira_idade.png",
    "Distribuições e fronteira de decisão - Idade"
)

grafico_fronteira(
    "duration",
    df["duration"].min(),
    df["duration"].max(),
    duracao[0],
    "fronteira_duracao.png",
    "Distribuições e fronteira de decisão - Duração"
)

print("\nGráficos de fronteira salvos em resultados/")