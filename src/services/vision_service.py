import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils.math_utils import normalize, stable_seed_from_image

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


def simulate_gradcam(image: Image.Image) -> Image.Image:
    """
    Simula Grad-CAM.

    Versão real futura:
        MRI -> ResNet50/CNN -> gradientes -> mapa de ativação.

    Versão atual:
        intensidade + foco anatômico central + overlay de heatmap.
    """
    image_rgb = image.convert("RGB").resize((512, 512))
    arr = np.array(image_rgb)

    gray = np.array(image_rgb.convert("L"))
    gray_norm = normalize(gray)

    h, w = gray_norm.shape
    yy, xx = np.ogrid[:h, :w]

    center_y = int(h * 0.52)
    center_x = int(w * 0.50)

    ellipse = (
        ((xx - center_x) ** 2) / (0.18 * w) ** 2
        + ((yy - center_y) ** 2) / (0.28 * h) ** 2
    )

    anatomical_focus = np.exp(-ellipse)
    heatmap = normalize(0.55 * gray_norm + 0.45 * anatomical_focus)

    if CV2_AVAILABLE:
        heat_uint8 = np.uint8(255 * heatmap)
        heat_uint8 = cv2.GaussianBlur(heat_uint8, (35, 35), 0)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(arr, 0.60, heat_color, 0.40, 0)
        return Image.fromarray(overlay)

    cmap = plt.get_cmap("jet")
    heat_color = np.uint8(255 * cmap(heatmap)[:, :, :3])
    overlay = np.uint8(0.60 * arr + 0.40 * heat_color)

    return Image.fromarray(overlay)


def simulate_anomaly_score(image: Image.Image) -> float:
    """
    Simula escore de anomalia visual.

    Versão real futura:
        MRI -> ResNet50 embeddings -> Isolation Forest -> anomaly score.

    Versão atual:
        usa contraste, brilho, entropia e assimetria visual para gerar
        um score mais variado e útil para demonstração.
    """
    image_gray = image.convert("L").resize((224, 224))
    arr = np.array(image_gray).astype(np.float32) / 255.0

    contrast = float(np.std(arr))
    brightness = float(np.mean(arr))

    hist, _ = np.histogram(arr, bins=32, range=(0, 1), density=True)
    hist = hist + 1e-8

    entropy = -np.sum(hist * np.log(hist))
    entropy_norm = float(entropy / np.log(32))

    # Mede diferença entre lado esquerdo e direito da imagem.
    left = arr[:, :112]
    right = arr[:, 112:]
    asymmetry = float(abs(np.mean(left) - np.mean(right)))

    # Mede presença de regiões muito claras/escuras.
    high_intensity_ratio = float(np.mean(arr > 0.75))
    low_intensity_ratio = float(np.mean(arr < 0.15))

    raw_score = (
        0.25 * contrast +
        0.25 * entropy_norm +
        0.20 * asymmetry +
        0.15 * high_intensity_ratio +
        0.15 * low_intensity_ratio
    )

    # Amplificação heurística para demonstração.
    score = raw_score * 1.8

    return float(np.clip(score, 0.05, 0.95))


def fuzzy_anomaly(score: float) -> dict:
    """
    Classificação fuzzy simplificada do escore de anomalia.

    Intervalos principais:
        0.00–0.33: Leve
        0.34–0.66: Moderado
        0.67–1.00: Severo
    """
    score = float(np.clip(score, 0, 1))

    leve = max(0.0, min(1.0, (0.50 - score) / 0.50))

    if score <= 0.25:
        moderado = score / 0.25
    elif score <= 0.50:
        moderado = 1.0
    elif score <= 0.75:
        moderado = (0.75 - score) / 0.25
    else:
        moderado = 0.0

    severo = max(0.0, min(1.0, (score - 0.50) / 0.50))

    memberships = {
        "Leve": float(leve),
        "Moderado": float(moderado),
        "Severo": float(severo),
    }

    # Regra mais objetiva para evitar sempre cair em Leve.
    if score < 0.34:
        label = "Leve"
    elif score < 0.67:
        label = "Moderado"
    else:
        label = "Severo"

    return {
        "label": label,
        "memberships": memberships,
    }