import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.math_utils import sigmoid


def asia_to_numeric(asia: str) -> float:
    return {
        "A": 0.00,
        "B": 0.25,
        "C": 0.50,
        "D": 0.75,
        "E": 1.00,
    }[asia]


def simulate_recovery_probability(
    asia: str,
    motor_strength: int,
    anomaly_score: float,
) -> float:
    """
    Simula predição clínica.

    Versão real futura:
        Random Forest/XGBoost treinado com dados longitudinais.

    Regra atual:
        ASIA melhor e força motora maior aumentam a recuperação.
        Anomalia visual maior reduz a recuperação.
    """
    asia_num = asia_to_numeric(asia)
    motor_norm = motor_strength / 100.0

    logit = -1.05 + 2.05 * asia_num + 2.25 * motor_norm - 1.85 * anomaly_score
    probability = sigmoid(logit)

    return float(np.clip(probability, 0.01, 0.99))


def simulate_shap_importance(
    asia: str,
    motor_strength: int,
    anomaly_score: float,
) -> pd.DataFrame:
    asia_num = asia_to_numeric(asia)
    motor_norm = motor_strength / 100.0

    weights = {
        "Força Motora": 0.45 + 0.20 * motor_norm,
        "Escala ASIA": 0.35 + 0.25 * asia_num,
        "Características da MRI": 0.25 + 0.40 * anomaly_score,
    }

    total = sum(weights.values())

    df = pd.DataFrame(
        [
            {
                "Variável": key,
                "Peso relativo": value / total,
            }
            for key, value in weights.items()
        ]
    )

    return df.sort_values("Peso relativo", ascending=True)


def get_risk_label(probability: float) -> str:
    if probability >= 0.70:
        return "Baixo"
    if probability >= 0.40:
        return "Intermediário"
    return "Alto"


def get_prognosis_label(probability: float) -> str:
    if probability >= 0.70:
        return "prognóstico motor favorável"
    if probability >= 0.40:
        return "prognóstico motor intermediário"
    return "prognóstico motor reservado"


def plot_shap_bar(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.barh(df["Variável"], df["Peso relativo"])
    ax.set_xlabel("Peso relativo na predição")
    ax.set_title("Explicabilidade Tabular — Importância Simulada tipo SHAP")
    ax.set_xlim(0, max(0.60, df["Peso relativo"].max() + 0.10))
    ax.grid(axis="x", alpha=0.25)

    for index, value in enumerate(df["Peso relativo"]):
        ax.text(value + 0.01, index, f"{value:.1%}", va="center")

    return fig


def plot_probability_gauge(probability: float):
    fig, ax = plt.subplots(figsize=(7, 3.8), subplot_kw={"projection": "polar"})

    theta = np.linspace(np.pi, 0, 120)
    ax.plot(theta, np.ones_like(theta), linewidth=18, solid_capstyle="round", alpha=0.25)

    filled_theta = np.linspace(np.pi, np.pi * (1 - probability), 120)
    ax.plot(filled_theta, np.ones_like(filled_theta), linewidth=18, solid_capstyle="round")

    pointer_theta = np.pi * (1 - probability)
    ax.plot([pointer_theta, pointer_theta], [0, 1], linewidth=3)

    ax.text(
        0,
        -0.15,
        f"{probability:.0%}",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.set_title("Probabilidade Simulada de Recuperação Motora", pad=20)
    ax.set_axis_off()

    return fig