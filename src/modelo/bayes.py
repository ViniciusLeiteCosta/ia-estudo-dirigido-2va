import numpy as np
from scipy.stats import norm, gamma


def likelihood(feature, valor, classe, p):
    if feature == "age":
        media, desvio = p["age"][classe]
        return norm.pdf(valor, media, desvio)

    if feature == "duration":
        return gamma.pdf(valor + 1, *p["duration"][classe])

    return p["marital"][classe][valor]


def posterior(feature, valor, p):
    probs = np.array([
        p["prior"][c] * likelihood(feature, valor, c, p)
        for c in (0, 1)
    ])

    return probs / probs.sum()


def log_scores(age, duration, marital, p):
    scores = []

    for c in (0, 1):
        media, desvio = p["age"][c]

        score = (
            np.log(p["prior"][c])
            + norm.logpdf(age, media, desvio)
            + gamma.logpdf(duration + 1, *p["duration"][c])
            + np.log(p["marital"][c][marital])
        )

        scores.append(score)

    return np.array(scores)


def posterior_nb(age, duration, marital, p):
    scores = log_scores(age, duration, marital, p)
    probs = np.exp(scores - scores.max())
    return probs / probs.sum()


def prever(age, duration, marital, p):
    return int(np.argmax(log_scores(age, duration, marital, p)))