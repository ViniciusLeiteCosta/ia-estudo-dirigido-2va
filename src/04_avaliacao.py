import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

from modelo.dados import preparar_dados
from modelo.bayes import prever

os.makedirs("resultados", exist_ok=True)

_, X_treino, X_teste, y_treino, y_teste, p = preparar_dados()

y_pred = [
    prever(linha.age, linha.duration, linha.marital, p)
    for linha in X_teste.itertuples()
]

cm = confusion_matrix(y_teste, y_pred)
tn, fp, fn, tp = cm.ravel()

acuracia = accuracy_score(y_teste, y_pred)
precisao = precision_score(y_teste, y_pred)
recall = recall_score(y_teste, y_pred)
f1 = f1_score(y_teste, y_pred)
baseline = (y_teste == 0).mean()

print("=== AVALIAÇÃO ===")

print("\n=== DIVISÃO DOS DADOS ===")
print("Treino: 80%")
print("Teste: 20%")
print("Semente: 42")
print(f"Treino: {len(X_treino)} registros")
print(f"Teste: {len(X_teste)} registros")

print("\nClasses no treino:")
for classe, qtd in y_treino.value_counts().sort_index().items():
    print(f"Classe {classe}: {qtd} ({qtd / len(y_treino):.2%})")

print("\nClasses no teste:")
for classe, qtd in y_teste.value_counts().sort_index().items():
    print(f"Classe {classe}: {qtd} ({qtd / len(y_teste):.2%})")

print("\n=== MATRIZ DE CONFUSÃO ===")
print(cm)

print(f"\nTN={tn} | FP={fp} | FN={fn} | TP={tp}")

print("\nInterpretação:")
print(f"TN: {tn} clientes não aderiram e foram previstos como não adesão.")
print(f"FP: {fp} clientes não aderiram, mas foram previstos como adesão.")
print(f"FN: {fn} clientes aderiram, mas foram previstos como não adesão.")
print(f"TP: {tp} clientes aderiram e foram previstos como adesão.")

print("\n=== MÉTRICAS ===")
print(f"Acurácia = {acuracia:.2%}")
print(f"Precisão = {precisao:.2%}")
print(f"Recall = {recall:.2%}")
print(f"F1-score = {f1:.6f}")
print(f"Baseline = {baseline:.2%}")

print(f"\nPositivos identificados: {tp} de {tp + fn}")

ConfusionMatrixDisplay(cm).plot()
plt.tight_layout()
plt.savefig("resultados/matriz_confusao.png")
plt.close()