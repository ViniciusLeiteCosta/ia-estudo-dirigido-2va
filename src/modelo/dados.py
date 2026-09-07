import pandas as pd
from scipy.stats import gamma
from sklearn.model_selection import train_test_split

ARQUIVO = "data/bank-full.csv"
FEATURES = ["age", "duration", "marital"]


def preparar_dados():
    df = pd.read_csv(ARQUIVO, sep=";")[FEATURES + ["y"]]
    df["y"] = df["y"].map({"no": 0, "yes": 1})

    X = df[FEATURES]
    y = df["y"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    treino = X_treino.copy()
    treino["y"] = y_treino

    parametros = {
        "prior": y_treino.value_counts(normalize=True).to_dict(),
        "age": {},
        "duration": {},
        "marital": {}
    }

    for classe in (0, 1):
        grupo = treino[treino["y"] == classe]

        parametros["age"][classe] = (
            grupo["age"].mean(),
            grupo["age"].std(ddof=0)
        )

        parametros["duration"][classe] = gamma.fit(
            grupo["duration"] + 1,
            floc=0
        )

        parametros["marital"][classe] = (
            grupo["marital"]
            .value_counts(normalize=True)
            .to_dict()
        )

    return df, X_treino, X_teste, y_treino, y_teste, parametros