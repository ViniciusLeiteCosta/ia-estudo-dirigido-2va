from modelo.dados import preparar_dados
from modelo.bayes import log_scores, posterior_nb, prever

_, _, _, _, _, p = preparar_dados()

print("=== NAIVE BAYES MANUAL ===")

exemplos = [
    (30, 60, "single"),
    (40, 300, "married"),
    (60, 1000, "divorced")
]

for age, duration, marital in exemplos:
    scores = log_scores(age, duration, marital, p)
    post = posterior_nb(age, duration, marital, p)

    print(
        f"\nIdade={age}, "
        f"Duração={duration}, "
        f"Marital={marital}"
    )

    print(f"Log-score 0 = {scores[0]:.6f}")
    print(f"Log-score 1 = {scores[1]:.6f}")

    print(
        f"Posterior: "
        f"0={post[0]:.2%} | "
        f"1={post[1]:.2%}"
    )

    print(
        f"Classe prevista = "
        f"{prever(age, duration, marital, p)}"
    )

print(
    "\nO modelo assume independência condicional "
    "entre age, duration e marital dado Y."
)