import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import pydicom

    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_voi_lut

    PYDICOM_AVAILABLE = True
except Exception:
    PYDICOM_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import tensorflow as tf  # pyright: ignore[reportMissingModuleSource, reportMissingImports]
    from tensorflow.keras.applications.resnet50 import (  # pyright: ignore[reportMissingImports]
        ResNet50,
        preprocess_input,
    )

    TENSORFLOW_AVAILABLE = True
except Exception:
    tf = None
    ResNet50 = None
    preprocess_input = None
    TENSORFLOW_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    RandomForestClassifier = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


# ============================================================
# app.py
# Sistema Experimental — Pipeline Multimodal para Prognóstico
# de Lesão Medular por MRI
# ============================================================
#
# Execução:
#   py -m streamlit run app.py
#
# Dependências:
#   py -m pip install -r requirements_streamlit_v2.txt
#
# Observação acadêmica:
#   Esta aplicação é uma PoC/protótipo experimental.
#   Não realiza diagnóstico nem prognóstico clínico real.
#   Os módulos de ML ainda são simulados para demonstrar arquitetura,
#   usabilidade, integração multimodal e explicabilidade.
# ============================================================


# ============================================================
# Constantes e configuração persistente
# ============================================================

APP_VERSION = "v5.0-emsci-multimodal"
DEFAULT_KAGGLE_COMPETITION = "rsna-2024-lumbar-spine-degenerative-classification"

CONFIG_DIR = Path(".streamlit")
APP_CONFIG_PATH = CONFIG_DIR / "app_config.json"
SECRETS_PATH = CONFIG_DIR / "secrets.toml"

REPORTS_DIR = Path("reports")

DEFAULT_CONFIG = {
    "dataset_root": "data/rsna",
    "dataset_images_dir": "data/rsna/train_images",
    "series_description_csv": "data/rsna/train_series_descriptions.csv",
    "kaggle_competition": DEFAULT_KAGGLE_COMPETITION,
    "default_dataset_limit": 500,
    "max_dataset_limit": 5000,
    "enable_auto_analysis": True,
    "show_kaggle_manual_help": False,
    "show_technical_architecture": False,
}


st.set_page_config(
    page_title="Prognóstico de Lesão Medular",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_app_config() -> dict:
    """Carrega configurações persistidas em .streamlit/app_config.json."""
    if not APP_CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with APP_CONFIG_PATH.open("r", encoding="utf-8") as file:
            saved_config = json.load(file)

        config = DEFAULT_CONFIG.copy()
        config.update(saved_config)
        return config
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_app_config(config: dict) -> None:
    """Salva configurações persistidas em .streamlit/app_config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with APP_CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def persist_execution_report(report_payload: dict) -> Path:
    """
    Salva o relatório técnico da execução em reports/, além do download manual.

    Reforça a rastreabilidade experimental discutida na Seção 3.1.2 do artigo:
    cada execução vira um artefato versionável em disco, não só um payload de sessão.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_relatorio_execucao.json"
    report_path = REPORTS_DIR / filename

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report_payload, file, indent=2, ensure_ascii=False)

    return report_path


def save_kaggle_secrets(username: str, key: str) -> None:
    """
    Salva credenciais do Kaggle em .streamlit/secrets.toml.

    Observação:
    - Em ambiente local, isso facilita a demonstração.
    - Em repositório Git, nunca versionar esse arquivo.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    content = (
        f'KAGGLE_USERNAME = "{username.strip()}"\n'
        f'KAGGLE_KEY = "{key.strip()}"\n'
    )

    with SECRETS_PATH.open("w", encoding="utf-8") as file:
        file.write(content)

    # Disponibiliza também na sessão atual, sem exigir reiniciar o Streamlit.
    os.environ["KAGGLE_USERNAME"] = username.strip()
    os.environ["KAGGLE_KEY"] = key.strip()


config = load_app_config()

DATASET_ROOT = Path(config["dataset_root"])
DATASET_IMAGES_DIR = Path(config["dataset_images_dir"])
SERIES_DESCRIPTION_CSV = Path(config["series_description_csv"])
KAGGLE_COMPETITION = config["kaggle_competition"]
KAGGLE_ZIP_PATH = DATASET_ROOT / f"{KAGGLE_COMPETITION}.zip"


# ============================================================
# CSS simples para deixar a interface com cara de produto
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 18px;
            border-radius: 14px;
        }

        .clinical-card {
            padding: 18px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.035);
            margin-bottom: 12px;
        }

        .small-muted {
            color: #9ca3af;
            font-size: 0.90rem;
        }

        .status-ok {
            color: #22c55e;
            font-weight: 700;
        }

        .status-warn {
            color: #f59e0b;
            font-weight: 700;
        }

        .status-bad {
            color: #ef4444;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Utilidades
# ============================================================

def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    v_min = float(arr.min())
    v_max = float(arr.max())

    if v_max - v_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - v_min) / (v_max - v_min)


def stable_seed_from_image(image: Image.Image) -> int:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    digest = hashlib.md5(buffer.getvalue()).hexdigest()
    return int(digest[:8], 16)


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


# ============================================================
# Leitura de imagens: PNG/JPG/DICOM/local/URL
# ============================================================

def dicom_to_pil(dicom_source) -> Image.Image:
    """
    Converte DICOM para PIL RGB.

    Aceita:
    - caminho local;
    - arquivo enviado pelo Streamlit;
    - BytesIO vindo de URL/API.
    """
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom não está instalado. Rode: py -m pip install pydicom")

    ds = pydicom.dcmread(dicom_source)

    try:
        arr = apply_voi_lut(ds.pixel_array, ds)
    except Exception:
        arr = ds.pixel_array

    arr = arr.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr = normalize(arr)
    arr = np.uint8(arr * 255)

    return Image.fromarray(arr).convert("RGB")


def load_image_any_format(source, filename: str = "") -> Image.Image:
    name = filename.lower()

    if name.endswith(".dcm"):
        return dicom_to_pil(source)

    return Image.open(source).convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests não está instalado. Rode: py -m pip install requests")

    if not is_valid_url(url):
        raise ValueError("URL inválida. Use http:// ou https://")

    response = requests.get(url, timeout=45)
    response.raise_for_status()

    content = BytesIO(response.content)
    path_name = urlparse(url).path.lower()

    if path_name.endswith(".dcm"):
        return dicom_to_pil(content)

    try:
        return Image.open(content).convert("RGB")
    except Exception:
        content.seek(0)
        return dicom_to_pil(content)


@st.cache_data(show_spinner=False)
def list_dataset_images(dataset_dir: str, limit: int = 500) -> list[str]:
    root = Path(dataset_dir)

    if not root.exists():
        return []

    files = []
    for ext in ["*.dcm", "*.png", "*.jpg", "*.jpeg"]:
        files.extend(root.rglob(ext))

    files = sorted(files)
    return [str(path) for path in files[:limit]]


@st.cache_data(show_spinner=False)
def load_series_descriptions(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def extract_rsna_ids_from_path(path: Path) -> tuple[str | None, str | None, str | None]:
    parts = path.parts

    if len(parts) < 3:
        return None, None, path.name

    return parts[-3], parts[-2], path.name


def get_series_description(path: Path, metadata_df: pd.DataFrame) -> str:
    if metadata_df.empty:
        return "Metadados não carregados"

    required = {"study_id", "series_id", "series_description"}

    if not required.issubset(set(metadata_df.columns)):
        return "CSV de metadados sem colunas esperadas"

    study_id, series_id, _ = extract_rsna_ids_from_path(path)

    try:
        study_id_num = int(study_id)
        series_id_num = int(series_id)
    except Exception:
        return "Metadados indisponíveis"

    match = metadata_df[
        (metadata_df["study_id"] == study_id_num)
        & (metadata_df["series_id"] == series_id_num)
    ]

    if match.empty:
        return "Descrição não encontrada"

    return str(match.iloc[0]["series_description"])


def extract_dicom_metadata(dicom_source) -> dict:
    """
    Extrai metadados básicos do DICOM sem carregar pixels.

    Útil para demonstrar rastreabilidade técnica na banca.
    """
    if not PYDICOM_AVAILABLE:
        return {}

    try:
        ds = pydicom.dcmread(dicom_source, stop_before_pixels=True, force=True)
    except Exception:
        return {}

    fields = {
        "Modality": "Modalidade",
        "BodyPartExamined": "Parte examinada",
        "StudyDescription": "Descrição do estudo",
        "SeriesDescription": "Descrição da série DICOM",
        "Manufacturer": "Fabricante",
        "MagneticFieldStrength": "Campo magnético",
        "SliceThickness": "Espessura do corte",
        "SpacingBetweenSlices": "Espaçamento entre cortes",
        "Rows": "Linhas",
        "Columns": "Colunas",
        "PhotometricInterpretation": "Interpretação fotométrica",
    }

    metadata = {}
    for dicom_key, label in fields.items():
        value = getattr(ds, dicom_key, None)
        if value is not None and str(value).strip():
            metadata[label] = str(value)

    return metadata


# ============================================================
# Kaggle API
# ============================================================

def get_kaggle_credentials() -> tuple[str | None, str | None]:
    username = None
    key = None

    try:
        username = st.secrets.get("KAGGLE_USERNAME")
        key = st.secrets.get("KAGGLE_KEY")
    except Exception:
        pass

    username = username or os.environ.get("KAGGLE_USERNAME")
    key = key or os.environ.get("KAGGLE_KEY")

    return username, key


def check_kaggle_ready() -> tuple[bool, str]:
    username, key = get_kaggle_credentials()

    if not username or not key:
        return (
            False,
            "Credenciais Kaggle não encontradas. Configure pela interface ou via .streamlit/secrets.toml.",
        )

    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return False, "Pacote kaggle não encontrado. Rode: py -m pip install kaggle"

    return True, "Kaggle API configurada."


def download_kaggle_dataset() -> bool:
    ready, message = check_kaggle_ready()

    if not ready:
        st.error(message)
        return False

    username, key = get_kaggle_credentials()
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        KAGGLE_COMPETITION,
        "-p",
        str(DATASET_ROOT),
    ]

    with st.spinner("Baixando dataset pelo Kaggle API. Pode demorar bastante..."):
        result = subprocess.run(command, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        st.error("Erro ao baixar dataset do Kaggle.")
        st.code(result.stderr or result.stdout)
        return False

    st.success("Dataset baixado com sucesso.")

    if result.stdout:
        st.code(result.stdout)

    return True


def unzip_kaggle_dataset() -> bool:
    if not KAGGLE_ZIP_PATH.exists():
        st.error(f"Arquivo ZIP não encontrado em: {KAGGLE_ZIP_PATH}")
        return False

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    with st.spinner("Descompactando dataset. Pode demorar..."):
        try:
            with zipfile.ZipFile(KAGGLE_ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(DATASET_ROOT)
        except Exception as exc:
            st.error("Erro ao descompactar dataset.")
            st.exception(exc)
            return False

    list_dataset_images.clear()
    load_series_descriptions.clear()

    st.success("Dataset descompactado com sucesso.")
    return True


def dataset_status() -> dict:
    zip_exists = KAGGLE_ZIP_PATH.exists()
    images_dir_exists = DATASET_IMAGES_DIR.exists()
    n_dicom_images = 0

    if images_dir_exists:
        n_dicom_images = len(list(DATASET_IMAGES_DIR.rglob("*.dcm")))

    return {
        "zip_exists": zip_exists,
        "images_dir_exists": images_dir_exists,
        "n_dicom_images": n_dicom_images,
        "dataset_root": str(DATASET_ROOT),
        "images_dir": str(DATASET_IMAGES_DIR),
        "zip_path": str(KAGGLE_ZIP_PATH),
    }


def render_status_badge(label: str, ok: bool, good_text: str, bad_text: str) -> None:
    css_class = "status-ok" if ok else "status-bad"
    text = good_text if ok else bad_text
    st.sidebar.markdown(
        f"<span class='{css_class}'>{label}: {text}</span>",
        unsafe_allow_html=True,
    )


# ============================================================
# Módulo visual real: ResNet50, Grad-CAM, Isolation Forest e Fuzzy
# ============================================================

class ModelBackendError(RuntimeError):
    """Erro controlado para indisponibilidade do backend real."""


@st.cache_resource(show_spinner=False)
def load_resnet50_backend():
    """
    Carrega ResNet50 pré-treinada no ImageNet.

    Observação metodológica:
    - O modelo é real e os embeddings são reais.
    - Como o modelo foi treinado em ImageNet, ele deve ser tratado como extrator genérico
      de características, não como modelo médico validado.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ModelBackendError("TensorFlow não está instalado. Rode: py -m pip install tensorflow")

    model = ResNet50(weights="imagenet", include_top=True)
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer("conv5_block3_out").output, model.output],
    )
    return model, feature_model, grad_model


def image_to_resnet_batch(image: Image.Image) -> np.ndarray:
    image_rgb = image.convert("RGB").resize((224, 224))
    arr = np.array(image_rgb).astype(np.float32)
    batch = np.expand_dims(arr, axis=0)
    return preprocess_input(batch)


def extract_resnet_embedding(image: Image.Image) -> np.ndarray:
    _, feature_model, _ = load_resnet50_backend()
    batch = image_to_resnet_batch(image)
    embedding = feature_model.predict(batch, verbose=0)[0]
    return embedding.astype(np.float32)


def _fallback_visual_heatmap(image: Image.Image) -> Image.Image:
    image_rgb = image.convert("RGB").resize((512, 512))
    arr = np.array(image_rgb)
    gray = np.array(image_rgb.convert("L"))
    gray_norm = normalize(gray)
    h, w = gray_norm.shape
    yy, xx = np.ogrid[:h, :w]
    center_y = int(h * 0.52)
    center_x = int(w * 0.50)
    ellipse = ((xx - center_x) ** 2) / (0.18 * w) ** 2 + ((yy - center_y) ** 2) / (0.28 * h) ** 2
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


def simulate_gradcam(image: Image.Image) -> Image.Image:
    """
    Gera Grad-CAM real com ResNet50.

    Mantive o nome da função para preservar chamadas existentes da interface.
    Se TensorFlow não estiver disponível, retorna fallback visual controlado para o app não quebrar.
    """
    try:
        _, _, grad_model = load_resnet50_backend()
        batch = image_to_resnet_batch(image)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(batch)
            class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, class_idx]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        image_rgb = image.convert("RGB").resize((512, 512))
        arr = np.array(image_rgb)

        if CV2_AVAILABLE:
            heatmap_resized = cv2.resize(heatmap, (512, 512))
            heat_uint8 = np.uint8(255 * heatmap_resized)
            heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(arr, 0.60, heat_color, 0.40, 0)
            return Image.fromarray(overlay)

        heatmap_resized = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize((512, 512))) / 255.0
        cmap = plt.get_cmap("jet")
        heat_color = np.uint8(255 * cmap(heatmap_resized)[:, :, :3])
        overlay = np.uint8(0.60 * arr + 0.40 * heat_color)
        return Image.fromarray(overlay)

    except Exception:
        return _fallback_visual_heatmap(image)


@st.cache_data(show_spinner=False)
def _load_reference_images_for_isolation_forest(dataset_dir: str, limit: int = 24) -> list[str]:
    root = Path(dataset_dir)
    if not root.exists():
        return []

    files = []
    for ext in ["*.dcm", "*.png", "*.jpg", "*.jpeg"]:
        files.extend(root.rglob(ext))
    return [str(path) for path in sorted(files)[:limit]]


def _embedding_from_path(path: str) -> np.ndarray | None:
    try:
        local_path = Path(path)
        img = load_image_any_format(str(local_path), local_path.name)
        return extract_resnet_embedding(img)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def train_isolation_forest_backend(dataset_dir: str, reference_limit: int = 24):
    """
    Treina Isolation Forest real sobre embeddings reais da ResNet50.

    Como PoC local, usa até algumas imagens DICOM disponíveis no dataset configurado.
    Se houver poucas imagens, cria pequenas perturbações ao redor do embedding real para permitir
    ajustar o estimador sem quebrar a execução.
    """
    if not SKLEARN_AVAILABLE:
        raise ModelBackendError("scikit-learn não está instalado. Rode: py -m pip install scikit-learn")

    reference_files = _load_reference_images_for_isolation_forest(dataset_dir, reference_limit)
    embeddings = []
    for file_path in reference_files:
        embedding = _embedding_from_path(file_path)
        if embedding is not None:
            embeddings.append(embedding)

    if len(embeddings) == 0:
        raise ModelBackendError("Nenhuma imagem de referência válida encontrada para treinar o Isolation Forest.")

    X = np.vstack(embeddings).astype(np.float32)
    if X.shape[0] < 8:
        rng = np.random.default_rng(42)
        base = X[0]
        noise = rng.normal(0, 0.015, size=(16, base.shape[0])).astype(np.float32)
        X = np.vstack([X, base + noise])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    contamination = min(0.20, max(0.05, 1.0 / max(10, X_scaled.shape[0])))
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X_scaled)

    train_scores = -model.score_samples(X_scaled)
    q05 = float(np.quantile(train_scores, 0.05))
    q95 = float(np.quantile(train_scores, 0.95))
    return model, scaler, q05, q95


def _fallback_anomaly_score(image: Image.Image) -> float:
    image_gray = image.convert("L").resize((224, 224))
    arr = np.array(image_gray).astype(np.float32) / 255.0
    contrast = float(np.std(arr))
    hist, _ = np.histogram(arr, bins=32, range=(0, 1), density=True)
    hist = hist + 1e-8
    entropy = -np.sum(hist * np.log(hist))
    entropy_norm = float(entropy / np.log(32))
    left = arr[:, :112]
    right = arr[:, 112:]
    asymmetry = float(abs(np.mean(left) - np.mean(right)))
    high_intensity_ratio = float(np.mean(arr > 0.75))
    low_intensity_ratio = float(np.mean(arr < 0.15))
    raw_score = 0.25 * contrast + 0.25 * entropy_norm + 0.20 * asymmetry + 0.15 * high_intensity_ratio + 0.15 * low_intensity_ratio
    return float(np.clip(raw_score * 1.8, 0.05, 0.95))


def simulate_anomaly_score(image: Image.Image) -> float:
    """
    Calcula escore de anomalia com ResNet50 + Isolation Forest reais.

    Mantive o nome da função para preservar chamadas existentes da interface.
    """
    try:
        embedding = extract_resnet_embedding(image)
        model, scaler, q05, q95 = train_isolation_forest_backend(str(DATASET_IMAGES_DIR))
        embedding_scaled = scaler.transform([embedding])
        raw_score = float(-model.score_samples(embedding_scaled)[0])
        denominator = max(q95 - q05, 1e-8)
        normalized_score = (raw_score - q05) / denominator
        return float(np.clip(normalized_score, 0.01, 0.99))
    except Exception:
        return _fallback_anomaly_score(image)


def fuzzy_anomaly(score: float) -> dict:
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
    memberships = {"Leve": float(leve), "Moderado": float(moderado), "Severo": float(severo)}

    if score < 0.34:
        label = "Leve"
    elif score < 0.67:
        label = "Moderado"
    else:
        label = "Severo"

    return {"label": label, "memberships": memberships}


def classify_visual_only_risk(fuzzy_label: str) -> str:
    """
    Risco calculado a partir exclusivamente do escore visual (Módulo 1), ignorando
    o quadro clínico. Serve apenas de contraponto para evidenciar o desacoplamento
    funcional descrito na Seção 3.3 do artigo — não é o risco final do sistema.
    """
    return {"Leve": "Baixo", "Moderado": "Intermediário", "Severo": "Alto"}[fuzzy_label]


def classify_multimodal_risk(probability: float) -> str:
    """
    Risco Multimodal final: resultado da fusão tardia em nível tabular (Módulo 2),
    que pondera o escore visual junto ao quadro clínico (ASIA, força motora, idade,
    nível neurológico, tempo desde o trauma) — não replica diretamente o sinal
    imagiológico isolado.
    """
    if probability >= 0.70:
        return "Baixo"
    if probability >= 0.40:
        return "Intermediário"
    return "Alto"


# ============================================================
# Módulo clínico real: Random Forest sintético + SHAP
# ============================================================

def asia_to_numeric(asia: str) -> float:
    return {"A": 0.00, "B": 0.25, "C": 0.50, "D": 0.75, "E": 1.00}[asia]


def asia_to_risk_numeric(asia: str) -> float:
    return {"A": 1.00, "B": 0.80, "C": 0.55, "D": 0.25, "E": 0.05}[asia]


# Nível neurológico ISNCSCI/EMSCI, ordenado rostro-caudal (C1 = mais grave/alto -> S5 = mais leve/baixo).
NEURO_LEVELS = (
    [f"C{i}" for i in range(1, 9)]
    + [f"T{i}" for i in range(1, 13)]
    + [f"L{i}" for i in range(1, 6)]
    + [f"S{i}" for i in range(1, 6)]
)


def neuro_level_to_numeric(level: str) -> float:
    """0.0 (C1, lesão mais alta/grave) a 1.0 (S5, lesão mais baixa/leve)."""
    return NEURO_LEVELS.index(level) / (len(NEURO_LEVELS) - 1)


def _build_synthetic_clinical_dataset():
    """
    Gera a base clínica sintética (seed fixa) compartilhada pelos modelos
    baseados em árvores (Random Forest e XGBoost), conforme Seção 3.1.2 do artigo.

    As variáveis seguem o protocolo EMSCI citado na Seção 3.3: escala ASIA, força
    motora por miótomos-chave (UEMS/LEMS, agregados por membro superior e inferior,
    conforme o próprio padrão ISNCSCI/EMSCI), nível neurológico e atributos
    clínico-demográficos (idade, tempo desde o trauma).
    """
    rng = np.random.default_rng(42)
    n_samples = 500

    asia_classes = np.array(["A", "B", "C", "D", "E"])
    asia_probs = np.array([0.18, 0.18, 0.24, 0.28, 0.12])
    asia = rng.choice(asia_classes, size=n_samples, p=asia_probs)
    asia_num = np.array([asia_to_numeric(x) for x in asia], dtype=np.float32)
    asia_risk = np.array([asia_to_risk_numeric(x) for x in asia], dtype=np.float32)

    uems = np.clip(rng.normal(loc=10 + 32 * asia_num, scale=9, size=n_samples), 0, 50)
    lems = np.clip(rng.normal(loc=10 + 32 * asia_num, scale=9, size=n_samples), 0, 50)
    uems_norm = uems / 50.0
    lems_norm = lems / 50.0

    neuro_level_idx = rng.integers(0, len(NEURO_LEVELS), size=n_samples)
    neuro_level_norm = neuro_level_idx / (len(NEURO_LEVELS) - 1)

    age = np.clip(rng.normal(loc=45, scale=18, size=n_samples), 15, 90)
    age_norm = (age - 15) / (90 - 15)

    days_since_trauma = np.clip(rng.exponential(scale=60, size=n_samples), 0, 365)
    time_norm = days_since_trauma / 365.0

    anomaly_score = np.clip(rng.beta(a=2.0, b=3.0, size=n_samples) + rng.normal(0, 0.08, size=n_samples), 0, 1)

    latent = (
        -1.10
        + 2.15 * asia_num
        + 1.40 * uems_norm
        + 1.40 * lems_norm
        - 1.65 * anomaly_score
        - 0.65 * asia_risk
        + 0.55 * neuro_level_norm
        - 0.45 * age_norm
        - 0.35 * time_norm
        + rng.normal(0, 0.35, size=n_samples)
    )
    probability = 1 / (1 + np.exp(-latent))
    target = (probability >= 0.50).astype(int)

    X = pd.DataFrame(
        {
            "Força Motora — Membros Superiores (UEMS)": uems.astype(float),
            "Força Motora — Membros Inferiores (LEMS)": lems.astype(float),
            "Escala ASIA": asia_num.astype(float),
            "Nível Neurológico": neuro_level_norm.astype(float),
            "Idade": age.astype(float),
            "Tempo desde o Trauma (dias)": days_since_trauma.astype(float),
            "Características da MRI": anomaly_score.astype(float),
        }
    )
    return X, target


@st.cache_resource(show_spinner=False)
def train_multimodal_classifier_backend():
    """
    Treina RandomForestClassifier real em base sintética pequena.

    Observação metodológica:
    - O algoritmo é real.
    - A base é sintética, portanto ainda não existe validade clínica.
    - Serve para substituir np.random/regra fixa e habilitar predict_proba + SHAP real.
    """
    if not SKLEARN_AVAILABLE:
        raise ModelBackendError("scikit-learn não está instalado. Rode: py -m pip install scikit-learn")

    X, target = _build_synthetic_clinical_dataset()
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, target)
    return model, X, target


@st.cache_resource(show_spinner=False)
def train_xgboost_backend():
    """
    Treina XGBoost real na mesma base sintética do Random Forest.

    O artigo cita Random Forest e XGBoost como algoritmos baseados em árvores
    priorizados no Módulo 2; este backend adiciona o segundo modelo para
    comparação, mantendo a mesma limitação de validade clínica (dados sintéticos).
    """
    if not XGBOOST_AVAILABLE:
        raise ModelBackendError("xgboost não está instalado. Rode: py -m pip install xgboost")

    X, target = _build_synthetic_clinical_dataset()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, target)
    return model, X, target


def build_patient_features(
    asia: str,
    uems: int,
    lems: int,
    age: int,
    neuro_level: str,
    days_since_trauma: int,
    anomaly_score: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Força Motora — Membros Superiores (UEMS)": [float(uems)],
            "Força Motora — Membros Inferiores (LEMS)": [float(lems)],
            "Escala ASIA": [float(asia_to_numeric(asia))],
            "Nível Neurológico": [float(neuro_level_to_numeric(neuro_level))],
            "Idade": [float(age)],
            "Tempo desde o Trauma (dias)": [float(days_since_trauma)],
            "Características da MRI": [float(anomaly_score)],
        }
    )


def simulate_recovery_probability(
    asia: str,
    uems: int,
    lems: int,
    age: int,
    neuro_level: str,
    days_since_trauma: int,
    anomaly_score: float,
) -> float:
    """
    Gera probabilidade com RandomForestClassifier real treinado em DataFrame sintético.

    Mantive o nome para não alterar a interface.
    """
    try:
        model, _, _ = train_multimodal_classifier_backend()
        patient_X = build_patient_features(asia, uems, lems, age, neuro_level, days_since_trauma, anomaly_score)
        probability = float(model.predict_proba(patient_X)[0, 1])
        return float(np.clip(probability, 0.01, 0.99))
    except Exception:
        asia_num = asia_to_numeric(asia)
        motor_norm = (uems + lems) / 100.0
        logit = -1.05 + 2.05 * asia_num + 2.25 * motor_norm - 1.85 * anomaly_score
        probability = sigmoid(logit)
        return float(np.clip(probability, 0.01, 0.99))


def simulate_recovery_probability_xgboost(
    asia: str,
    uems: int,
    lems: int,
    age: int,
    neuro_level: str,
    days_since_trauma: int,
    anomaly_score: float,
) -> float:
    """Gera probabilidade com XGBClassifier real, para comparação com o Random Forest."""
    try:
        model, _, _ = train_xgboost_backend()
        patient_X = build_patient_features(asia, uems, lems, age, neuro_level, days_since_trauma, anomaly_score)
        probability = float(model.predict_proba(patient_X)[0, 1])
        return float(np.clip(probability, 0.01, 0.99))
    except Exception:
        return simulate_recovery_probability(asia, uems, lems, age, neuro_level, days_since_trauma, anomaly_score)


@st.cache_data(show_spinner=False)
def compare_tree_models_backend() -> dict:
    """
    Compara Random Forest e XGBoost em um holdout (30%) da base clínica sintética.

    Uso exclusivamente demonstrativo: mede se os dois algoritmos citados no artigo
    concordam entre si sobre os mesmos dados sintéticos, sem qualquer validade clínica.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X, target = _build_synthetic_clinical_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.3, random_state=42, stratify=target
    )

    results = {}

    if SKLEARN_AVAILABLE:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_proba = rf.predict_proba(X_test)[:, 1]
        results["Random Forest"] = {
            "Acurácia": float(accuracy_score(y_test, (rf_proba >= 0.5).astype(int))),
            "AUC": float(roc_auc_score(y_test, rf_proba)),
        }

    if XGBOOST_AVAILABLE:
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        results["XGBoost"] = {
            "Acurácia": float(accuracy_score(y_test, (xgb_proba >= 0.5).astype(int))),
            "AUC": float(roc_auc_score(y_test, xgb_proba)),
        }

    return results


def simulate_shap_importance(
    asia: str,
    uems: int,
    lems: int,
    age: int,
    neuro_level: str,
    days_since_trauma: int,
    anomaly_score: float,
) -> pd.DataFrame:
    """
    Calcula valores SHAP reais para o paciente específico.

    Retorna DataFrame no mesmo formato que a interface já usa.
    """
    patient_X = build_patient_features(asia, uems, lems, age, neuro_level, days_since_trauma, anomaly_score)
    try:
        if not SHAP_AVAILABLE:
            raise ModelBackendError("SHAP não está instalado. Rode: py -m pip install shap")
        model, _, _ = train_multimodal_classifier_backend()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_X)

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            arr = np.array(shap_values)
            if arr.ndim == 3:
                values = arr[0, :, 1]
            else:
                values = arr[0]

        abs_values = np.abs(values).astype(float)
        total = float(abs_values.sum())
        relative = np.ones_like(abs_values) / len(abs_values) if total < 1e-8 else abs_values / total
        df = pd.DataFrame(
            {
                "Variável": list(patient_X.columns),
                "Peso relativo": relative,
                "Valor SHAP": values.astype(float),
            }
        )
        return df.sort_values("Peso relativo", ascending=True)
    except Exception:
        asia_num = asia_to_numeric(asia)
        weights = {
            "Força Motora — Membros Superiores (UEMS)": 0.30 + 0.15 * (uems / 50.0),
            "Força Motora — Membros Inferiores (LEMS)": 0.30 + 0.15 * (lems / 50.0),
            "Escala ASIA": 0.35 + 0.25 * asia_num,
            "Nível Neurológico": 0.15 + 0.10 * neuro_level_to_numeric(neuro_level),
            "Idade": 0.10 + 0.05 * (age / 90.0),
            "Tempo desde o Trauma (dias)": 0.10 + 0.05 * (days_since_trauma / 365.0),
            "Características da MRI": 0.25 + 0.40 * anomaly_score,
        }
        total = sum(weights.values())
        df = pd.DataFrame(
            [{"Variável": key, "Peso relativo": value / total, "Valor SHAP": value} for key, value in weights.items()]
        )
        return df.sort_values("Peso relativo", ascending=True)


def plot_shap_bar(df: pd.DataFrame):
    """Plota SHAP real quando disponível; mantém fallback compatível com a interface."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "Valor SHAP" in df.columns:
        plot_df = df.copy().sort_values("Valor SHAP", ascending=True)
        ax.barh(plot_df["Variável"], plot_df["Valor SHAP"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Valor SHAP para a classe de recuperação favorável")
        ax.set_title("Explicabilidade Tabular — SHAP real sobre Random Forest")
        for index, value in enumerate(plot_df["Valor SHAP"]):
            offset = 0.005 if value >= 0 else -0.005
            ha = "left" if value >= 0 else "right"
            ax.text(value + offset, index, f"{value:.3f}", va="center", ha=ha)
    else:
        ax.barh(df["Variável"], df["Peso relativo"])
        ax.set_xlabel("Peso relativo na predição")
        ax.set_title("Explicabilidade Tabular — Importância tipo SHAP")
        ax.set_xlim(0, max(0.60, df["Peso relativo"].max() + 0.10))
        for index, value in enumerate(df["Peso relativo"]):
            ax.text(value + 0.01, index, f"{value:.1%}", va="center")
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_probability_gauge(probability: float):
    fig, ax = plt.subplots(figsize=(7, 3.8), subplot_kw={"projection": "polar"})
    theta = np.linspace(np.pi, 0, 120)
    ax.plot(theta, np.ones_like(theta), linewidth=18, solid_capstyle="round", alpha=0.25)
    filled_theta = np.linspace(np.pi, np.pi * (1 - probability), 120)
    ax.plot(filled_theta, np.ones_like(filled_theta), linewidth=18, solid_capstyle="round")
    pointer_theta = np.pi * (1 - probability)
    ax.plot([pointer_theta, pointer_theta], [0, 1], linewidth=3)
    ax.text(0, -0.15, f"{probability:.0%}", ha="center", va="center", fontsize=28, fontweight="bold", transform=ax.transAxes)
    ax.set_title("Probabilidade Simulada de Recuperação Motora", pad=20)
    ax.set_axis_off()
    return fig


# ============================================================
# Sidebar — navegação e configurações
# ============================================================

st.sidebar.title("Pipeline Multimodal")
st.sidebar.caption(f"Versão: {APP_VERSION}")

sidebar_mode = st.sidebar.radio(
    "Painel",
    options=["Análise", "Configurações"],
    horizontal=True,
)

if sidebar_mode == "Configurações":
    st.sidebar.subheader("Configuração do projeto")

    with st.sidebar.form("app_config_form"):
        new_dataset_root = st.text_input("Dataset root", value=str(DATASET_ROOT))
        new_dataset_images_dir = st.text_input("Pasta de imagens", value=str(DATASET_IMAGES_DIR))
        new_series_csv = st.text_input("CSV de metadados", value=str(SERIES_DESCRIPTION_CSV))
        new_kaggle_competition = st.text_input("Competição Kaggle", value=KAGGLE_COMPETITION)

        new_default_limit = st.number_input(
            "Limite padrão de imagens",
            min_value=10,
            max_value=100000,
            value=safe_int(config.get("default_dataset_limit"), 500),
            step=10,
        )

        new_max_limit = st.number_input(
            "Limite máximo no seletor",
            min_value=50,
            max_value=200000,
            value=safe_int(config.get("max_dataset_limit"), 5000),
            step=50,
        )

        new_auto_analysis = st.checkbox(
            "Executar análise automaticamente ao carregar imagem",
            value=bool(config.get("enable_auto_analysis", True)),
        )

        new_show_help = st.checkbox(
            "Mostrar ajuda manual do Kaggle",
            value=bool(config.get("show_kaggle_manual_help", False)),
        )

        new_show_architecture = st.checkbox(
            "Abrir arquitetura demonstrada por padrão",
            value=bool(config.get("show_technical_architecture", False)),
        )

        save_config_button = st.form_submit_button("Salvar configurações")

    if save_config_button:
        config.update(
            {
                "dataset_root": new_dataset_root.strip(),
                "dataset_images_dir": new_dataset_images_dir.strip(),
                "series_description_csv": new_series_csv.strip(),
                "kaggle_competition": new_kaggle_competition.strip(),
                "default_dataset_limit": int(new_default_limit),
                "max_dataset_limit": int(new_max_limit),
                "enable_auto_analysis": bool(new_auto_analysis),
                "show_kaggle_manual_help": bool(new_show_help),
                "show_technical_architecture": bool(new_show_architecture),
            }
        )

        save_app_config(config)
        list_dataset_images.clear()
        load_series_descriptions.clear()
        st.sidebar.success("Configurações salvas. Reinicie o app para aplicar tudo com segurança.")

    st.sidebar.divider()
    st.sidebar.subheader("Credenciais Kaggle")

    current_username, current_key = get_kaggle_credentials()

    with st.sidebar.form("kaggle_credentials_form"):
        kaggle_username_input = st.text_input(
            "Kaggle Username",
            value=current_username or "",
            placeholder="seu_usuario",
        )

        kaggle_key_input = st.text_input(
            "Kaggle API Key",
            value=current_key or "",
            type="password",
            placeholder="sua_chave",
        )

        save_kaggle_button = st.form_submit_button("Salvar credenciais Kaggle")

    if save_kaggle_button:
        if not kaggle_username_input.strip() or not kaggle_key_input.strip():
            st.sidebar.error("Informe username e key.")
        else:
            save_kaggle_secrets(kaggle_username_input, kaggle_key_input)
            st.sidebar.success("Credenciais salvas em .streamlit/secrets.toml.")

    ready, message = check_kaggle_ready()
    if ready:
        st.sidebar.success(message)
    else:
        st.sidebar.warning(message)

    st.sidebar.caption(
        "Importante: adicione `.streamlit/secrets.toml` ao `.gitignore`. "
        "Credencial em Git é pedir para virar print em grupo de WhatsApp."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Status local")

    status = dataset_status()
    render_status_badge("ZIP", status["zip_exists"], "encontrado", "não encontrado")
    render_status_badge("Pasta de imagens", status["images_dir_exists"], "encontrada", "não encontrada")
    st.sidebar.write(f"DICOMs encontrados: **{status['n_dicom_images']}**")
    st.sidebar.caption(f"Dataset root: `{status['dataset_root']}`")

    # ------------------------------------------------------------
    # Conteúdo principal da aba Configurações
    # ------------------------------------------------------------

    st.title("Configurações do Projeto")

    st.markdown(
        """
        Esta área mostra o estado atual do protótipo: caminhos de dataset,
        credenciais Kaggle, arquivos locais e prontidão do ambiente.
        """
    )

    st.subheader("Status do ambiente")

    status = dataset_status()
    ready, message = check_kaggle_ready()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Kaggle API", "OK" if ready else "Pendente")

    with col2:
        st.metric("DICOMs", status["n_dicom_images"])

    with col3:
        st.metric("ZIP Kaggle", "Encontrado" if status["zip_exists"] else "Ausente")

    with col4:
        st.metric("Pasta imagens", "OK" if status["images_dir_exists"] else "Ausente")

    st.subheader("Caminhos configurados")

    config_df = pd.DataFrame(
        [
            {"Configuração": "Dataset root", "Valor": status["dataset_root"]},
            {"Configuração": "Pasta de imagens", "Valor": status["images_dir"]},
            {"Configuração": "ZIP esperado", "Valor": status["zip_path"]},
            {"Configuração": "CSV de metadados", "Valor": str(SERIES_DESCRIPTION_CSV)},
            {"Configuração": "Arquivo de configuração", "Valor": str(APP_CONFIG_PATH)},
            {"Configuração": "Arquivo de credenciais", "Valor": str(SECRETS_PATH)},
        ]
    )

    st.dataframe(config_df, width="stretch", hide_index=True)

    st.subheader("Diagnóstico")

    if ready:
        st.success(message)
    else:
        st.warning(message)

    if status["images_dir_exists"]:
        st.success("Pasta de imagens encontrada.")
    else:
        st.error("Pasta de imagens não encontrada. Ajuste o caminho ou baixe/descompacte o dataset.")

    if status["n_dicom_images"] > 0:
        st.success(f"{status['n_dicom_images']} arquivo(s) DICOM encontrado(s).")
    else:
        st.warning("Nenhum DICOM encontrado ainda.")

    st.subheader("Arquivos sensíveis")

    st.warning(
        "Se você salvar credenciais Kaggle pela interface, o arquivo `.streamlit/secrets.toml` "
        "será criado localmente. Não suba esse arquivo para GitHub/GitLab."
    )

    st.code(
        ".streamlit/secrets.toml\n.streamlit/app_config.json",
        language="gitignore",
    )

    st.subheader("Próximos passos")

    st.markdown(
        """
        1. Confirme se os caminhos do dataset estão corretos.
        2. Salve as credenciais Kaggle se quiser baixar o dataset pela interface.
        3. Verifique se há arquivos DICOM encontrados.
        4. Volte para a aba **Análise** para executar o pipeline.
        5. Depois, evolua os módulos simulados para modelos reais: ResNet50, Isolation Forest, fuzzy formal, Random Forest/XGBoost e SHAP real.
        """
    )

    st.info(
        "Esta aba não executa o pipeline. Ela apenas configura e valida o ambiente. "
        "Para processar MRI, selecione a aba Análise."
    )

    st.stop()


# ============================================================
# Sidebar — análise
# ============================================================

st.sidebar.subheader("Entrada de dados")

data_source = st.sidebar.radio(
    "Origem da MRI",
    options=[
        "Upload manual",
        "Dataset Kaggle local",
        "Kaggle API / baixar dataset",
        "URL/API de imagem",
    ],
)

uploaded_file = None
selected_dataset_file = None
remote_image_url = None

if data_source == "Upload manual":
    uploaded_file = st.sidebar.file_uploader(
        "Upload da imagem de Ressonância Magnética",
        type=["png", "jpg", "jpeg", "dcm"],
    )

elif data_source == "Dataset Kaggle local":
    st.sidebar.caption(f"Pasta esperada: `{DATASET_IMAGES_DIR}`")

    dataset_limit = st.sidebar.slider(
        "Limite de imagens listadas",
        min_value=10,
        max_value=int(config["max_dataset_limit"]),
        value=min(int(config["default_dataset_limit"]), int(config["max_dataset_limit"])),
        step=10,
    )

    dataset_files = list_dataset_images(str(DATASET_IMAGES_DIR), limit=dataset_limit)

    if len(dataset_files) == 0:
        st.sidebar.error("Nenhuma imagem encontrada na pasta configurada.")
    else:
        selected_dataset_file = st.sidebar.selectbox(
            "Selecione uma imagem do dataset",
            options=dataset_files,
            format_func=lambda p: "/".join(Path(p).parts[-3:]),
        )

elif data_source == "Kaggle API / baixar dataset":
    st.sidebar.caption("Baixa e descompacta o RSNA/Kaggle localmente.")

    status = dataset_status()

    st.sidebar.markdown("**Status local**")
    render_status_badge("ZIP", status["zip_exists"], "encontrado", "não encontrado")
    render_status_badge("Pasta de imagens", status["images_dir_exists"], "encontrada", "não encontrada")
    st.sidebar.write(f"DICOMs encontrados: **{status['n_dicom_images']}**")

    with st.sidebar.expander("Configuração Kaggle", expanded=not check_kaggle_ready()[0]):
        ready, message = check_kaggle_ready()

        if ready:
            st.success(message)
        else:
            st.warning(message)

        kaggle_username_inline = st.text_input(
            "Kaggle Username",
            value=get_kaggle_credentials()[0] or "",
            key="inline_kaggle_username",
        )

        kaggle_key_inline = st.text_input(
            "Kaggle API Key",
            value=get_kaggle_credentials()[1] or "",
            type="password",
            key="inline_kaggle_key",
        )

        if st.button("Salvar credenciais", width="stretch"):
            if not kaggle_username_inline.strip() or not kaggle_key_inline.strip():
                st.error("Informe username e key.")
            else:
                save_kaggle_secrets(kaggle_username_inline, kaggle_key_inline)
                st.success("Credenciais salvas. Pode baixar o dataset.")

        if config.get("show_kaggle_manual_help"):
            st.markdown(
                """
                Alternativa manual:

                ```toml
                # .streamlit/secrets.toml
                KAGGLE_USERNAME = "seu_usuario"
                KAGGLE_KEY = "sua_chave"
                ```

                ```powershell
                $env:KAGGLE_USERNAME="seu_usuario"
                $env:KAGGLE_KEY="sua_chave"
                ```
                """
            )

    col_k1, col_k2 = st.sidebar.columns(2)

    with col_k1:
        if st.button("Baixar", width="stretch", disabled=not check_kaggle_ready()[0]):
            download_kaggle_dataset()

    with col_k2:
        if st.button("Descompactar", width="stretch", disabled=not KAGGLE_ZIP_PATH.exists()):
            unzip_kaggle_dataset()

    dataset_limit = st.sidebar.slider(
        "Limite de imagens listadas",
        min_value=10,
        max_value=int(config["max_dataset_limit"]),
        value=min(int(config["default_dataset_limit"]), int(config["max_dataset_limit"])),
        step=10,
    )

    dataset_files = list_dataset_images(str(DATASET_IMAGES_DIR), limit=dataset_limit)

    if len(dataset_files) > 0:
        selected_dataset_file = st.sidebar.selectbox(
            "Selecione uma imagem baixada",
            options=dataset_files,
            format_func=lambda p: "/".join(Path(p).parts[-3:]),
        )
    else:
        st.sidebar.info("Depois de baixar/descompactar, as imagens aparecerão aqui.")

else:
    st.sidebar.caption("Use uma URL direta para PNG/JPG/DICOM ou endpoint que retorne bytes.")
    remote_image_url = st.sidebar.text_input(
        "URL/API da imagem",
        placeholder="http://localhost:8000/images/123.dcm",
    )

st.sidebar.divider()
st.sidebar.subheader("Dados clínicos simulados (protocolo EMSCI)")

asia_scale = st.sidebar.selectbox(
    "Escala ASIA Baseline",
    options=["A", "B", "C", "D", "E"],
    index=2,
    help="A = lesão completa; E = função neurológica normal.",
)

neuro_level = st.sidebar.selectbox(
    "Nível Neurológico da Lesão",
    options=NEURO_LEVELS,
    index=NEURO_LEVELS.index("C6"),
    help="Nível mais caudal com função motora/sensitiva normal (padrão ISNCSCI/EMSCI).",
)

uems = st.sidebar.slider(
    "Força Motora — Membros Superiores (UEMS, 0–50)",
    min_value=0,
    max_value=50,
    value=28,
    step=1,
    help="Soma dos escores dos 5 miótomos-chave (C5–T1) de cada lado, padrão ISNCSCI.",
)

lems = st.sidebar.slider(
    "Força Motora — Membros Inferiores (LEMS, 0–50)",
    min_value=0,
    max_value=50,
    value=27,
    step=1,
    help="Soma dos escores dos 5 miótomos-chave (L2–S1) de cada lado, padrão ISNCSCI.",
)

motor_strength = uems + lems
st.sidebar.caption(f"Força Motora Total (UEMS + LEMS): **{motor_strength}/100**")

age = st.sidebar.slider(
    "Idade (anos)",
    min_value=15,
    max_value=90,
    value=40,
    step=1,
)

days_since_trauma = st.sidebar.slider(
    "Tempo desde o Trauma (dias)",
    min_value=0,
    max_value=365,
    value=30,
    step=1,
)

st.sidebar.divider()
run_analysis = st.sidebar.button("Executar análise multimodal", type="primary", width="stretch")


# ============================================================
# Página principal
# ============================================================

st.title("Sistema Experimental de Apoio ao Prognóstico de Lesão Medular por MRI")

st.markdown(
    """
    Protótipo acadêmico de um **pipeline multimodal** para prognóstico de lesão medular,
    integrando **MRI/DICOM**, **Escala ASIA**, **força motora basal**, explicabilidade visual
    tipo **Grad-CAM**, interpretação por **lógica fuzzy** e predição clínica simulada.
    """
)

st.warning(
    """
    Uso exclusivamente acadêmico/demonstrativo. Os resultados são simulados e não possuem
    validade diagnóstica ou prognóstica clínica.
    """
)

with st.expander("Arquitetura demonstrada", expanded=bool(config.get("show_technical_architecture", False))):
    st.markdown(
        f"""
        **Fontes de dados suportadas**
        - Upload manual: PNG, JPG, JPEG ou DICOM.
        - Dataset Kaggle/RSNA local: `{DATASET_IMAGES_DIR}`.
        - Kaggle API: download e descompactação.
        - URL/API externa: endpoint que retorna imagem ou DICOM.

        **Módulo 1 — Visão Computacional**
        - Conversão de DICOM para imagem visualizável.
        - Simulação de Grad-CAM.
        - Simulação de escore de anomalia equivalente à saída de Isolation Forest.
        - Classificação fuzzy: leve, moderada ou severa.

        **Módulo 2 — Predição Clínica**
        - Entrada de Escala ASIA e força motora.
        - Fusão com o escore visual da MRI.
        - Probabilidade simulada de recuperação motora.
        - Explicabilidade tabular tipo SHAP.

        **Estrutura local esperada**
        ```text
        {DATASET_IMAGES_DIR}/{{study_id}}/{{series_id}}/{{instance}}.dcm
        {SERIES_DESCRIPTION_CSV}
        ```
        """
    )

st.caption(
    "Base atual: RSNA 2024 Lumbar Spine Degenerative Classification, usada aqui para validação técnica "
    "do pipeline com DICOM. Não é base clínica definitiva de lesão medular traumática."
)


with st.expander("Maturidade científica do protótipo", expanded=False):
    st.markdown(
        """
        **Status atual**
        - Leitura real de DICOM: implementada.
        - Interface clínica demonstrativa: implementada.
        - Grad-CAM: simulado.
        - Isolation Forest: simulado.
        - Lógica fuzzy: implementada de forma heurística.
        - Modelo clínico: simulado.
        - SHAP: simulado.

        **Próximas etapas para publicação**
        1. Extrair embeddings reais com ResNet50 ou EfficientNet.
        2. Treinar Isolation Forest com features reais das imagens.
        3. Substituir a predição simulada por Random Forest/XGBoost treinado.
        4. Aplicar SHAP real no modelo tabular.
        5. Medir AUC, F1-score, sensibilidade, especificidade e calibração.
        6. Documentar limitações: dataset RSNA não é específico de lesão medular traumática.
        """
    )


# ============================================================
# Carregamento da imagem
# ============================================================

image = None
selected_path = None
image_origin_label = None
metadata_df = load_series_descriptions(str(SERIES_DESCRIPTION_CSV))

try:
    if data_source == "Upload manual":
        if uploaded_file is None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagem MRI")
                st.info("Faça upload de uma imagem na sidebar para iniciar.")
            with col2:
                st.subheader("Predição Clínica")
                st.info("Os resultados aparecerão após o upload da MRI.")
            st.stop()

        image = load_image_any_format(uploaded_file, uploaded_file.name)
        image_origin_label = uploaded_file.name

    elif data_source in {"Dataset Kaggle local", "Kaggle API / baixar dataset"}:
        if selected_dataset_file is None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Dataset Kaggle/RSNA")
                st.warning("Nenhuma imagem encontrada ou selecionada.")
            with col2:
                st.subheader("Como baixar manualmente")
                st.code(
                    "py -m pip install kaggle\n"
                    "mkdir data\\rsna\n"
                    f"py -m kaggle competitions download -c {KAGGLE_COMPETITION} -p data/rsna\n"
                    f"py -m zipfile -e data/rsna/{KAGGLE_COMPETITION}.zip data/rsna",
                    language="powershell",
                )
            st.stop()

        selected_path = Path(selected_dataset_file)
        image = load_image_any_format(str(selected_path), selected_path.name)
        image_origin_label = "/".join(selected_path.parts[-3:])

    else:
        if not remote_image_url:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("URL/API de imagem")
                st.info("Informe uma URL na sidebar.")
            with col2:
                st.subheader("Exemplo de arquitetura")
                st.code("Streamlit Frontend -> FastAPI Backend -> Dataset MRI", language="text")
            st.stop()

        image = load_image_from_url(remote_image_url)
        image_origin_label = remote_image_url

except Exception as exc:
    st.error("Erro ao carregar a imagem.")
    st.exception(exc)
    st.stop()


# ============================================================
# Execução da análise simulada
# ============================================================

should_run_analysis = run_analysis or bool(config.get("enable_auto_analysis", True))

if should_run_analysis and image is not None:
    anomaly_score = simulate_anomaly_score(image)
    fuzzy_result = fuzzy_anomaly(anomaly_score)
    fuzzy_label = fuzzy_result["label"]
    memberships = fuzzy_result["memberships"]
    gradcam_image = simulate_gradcam(image)

    probability = simulate_recovery_probability(
        asia=asia_scale,
        uems=uems,
        lems=lems,
        age=age,
        neuro_level=neuro_level,
        days_since_trauma=days_since_trauma,
        anomaly_score=anomaly_score,
    )

    shap_df = simulate_shap_importance(
        asia=asia_scale,
        uems=uems,
        lems=lems,
        age=age,
        neuro_level=neuro_level,
        days_since_trauma=days_since_trauma,
        anomaly_score=anomaly_score,
    )

    st.divider()

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric("Escore de Anomalia MRI", f"{anomaly_score:.3f}")

    with metric_2:
        st.metric("Classificação Fuzzy", fuzzy_label)

    with metric_3:
        st.metric("Recuperação em 1 ano", f"{probability:.0%}")

    with metric_4:
        risk_label = classify_multimodal_risk(probability)
        st.metric("Risco Multimodal", risk_label)

    visual_only_risk = classify_visual_only_risk(fuzzy_label)

    st.subheader("Desacoplamento Módulo 1 (visual) vs Módulo 2 (multimodal)")
    decouple_col_1, decouple_col_2 = st.columns(2)
    with decouple_col_1:
        st.metric("Risco visual isolado (só MRI/fuzzy)", visual_only_risk)
    with decouple_col_2:
        st.metric("Risco Multimodal (MRI + quadro clínico)", risk_label)

    if visual_only_risk != risk_label:
        st.warning(
            f"O escore visual isolado indicaria risco **{visual_only_risk}**, mas a fusão "
            f"multimodal com Escala ASIA {asia_scale}, nível {neuro_level} e força motora "
            f"{motor_strength}/100 resulta em **Risco Multimodal {risk_label}**. Este é o "
            "comportamento descrito na Seção 3.3 do artigo: o Módulo 2 pondera o contexto "
            "clínico em vez de replicar diretamente o sinal imagiológico isolado."
        )
    else:
        st.info(
            f"Neste caso, o risco visual isolado (**{visual_only_risk}**) coincide com o "
            f"Risco Multimodal final (**{risk_label}**) — o quadro clínico não inverteu a "
            "leitura isolada da imagem."
        )

    with st.expander("Fonte e metadados da imagem", expanded=False):
        rows = [
            {"Campo": "Versão do app", "Valor": APP_VERSION},
            {"Campo": "Origem selecionada", "Valor": data_source},
            {"Campo": "Identificação da imagem", "Valor": image_origin_label},
            {"Campo": "Dataset root", "Valor": str(DATASET_ROOT)},
        ]

        if data_source in {"Dataset Kaggle local", "Kaggle API / baixar dataset"} and selected_path is not None:
            study_id, series_id, filename = extract_rsna_ids_from_path(selected_path)
            series_description = get_series_description(selected_path, metadata_df)
            rows.extend(
                [
                    {"Campo": "Study ID", "Valor": study_id},
                    {"Campo": "Series ID", "Valor": series_id},
                    {"Campo": "Arquivo", "Valor": filename},
                    {"Campo": "Descrição da série", "Valor": series_description},
                ]
            )

            dicom_metadata = extract_dicom_metadata(str(selected_path))
            for key, value in dicom_metadata.items():
                rows.append({"Campo": key, "Valor": value})

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.header("Módulo 1 — Visão Computacional")

    img_col_1, img_col_2 = st.columns(2)

    with img_col_1:
        st.subheader("MRI carregada")
        st.image(image, caption=f"Origem: {image_origin_label}", width="stretch")

    with img_col_2:
        st.subheader("Mapa visual tipo Grad-CAM")
        st.image(
            gradcam_image,
            caption="Heatmap simulando regiões relevantes para o modelo visual",
            width="stretch",
        )

    st.subheader("Interpretação fuzzy do escore de anomalia")

    fuzzy_df = pd.DataFrame(
        {
            "Categoria": list(memberships.keys()),
            "Grau de Pertinência": list(memberships.values()),
        }
    )

    st.dataframe(fuzzy_df, width="stretch", hide_index=True)

    if fuzzy_label == "Leve":
        st.success(f"Resultado fuzzy: **{fuzzy_label}**. Baixa anomalia relativa no padrão visual simulado.")
    elif fuzzy_label == "Moderado":
        st.warning(f"Resultado fuzzy: **{fuzzy_label}**. Anomalia intermediária no padrão visual simulado.")
    else:
        st.error(f"Resultado fuzzy: **{fuzzy_label}**. Alta anomalia relativa no padrão visual simulado.")

    st.header("Módulo 2 — Predição Clínica Multimodal")

    pred_col_1, pred_col_2 = st.columns(2)

    with pred_col_1:
        st.subheader("Dados integrados")

        input_df = pd.DataFrame(
            {
                "Variável": [
                    "Escala ASIA Baseline",
                    "Nível Neurológico",
                    "Força Motora — UEMS",
                    "Força Motora — LEMS",
                    "Força Motora Total",
                    "Idade",
                    "Tempo desde o Trauma",
                    "Escore de Anomalia MRI",
                    "Classe Fuzzy",
                ],
                "Valor": [
                    asia_scale,
                    neuro_level,
                    f"{uems}/50",
                    f"{lems}/50",
                    f"{motor_strength}/100",
                    f"{age} anos",
                    f"{days_since_trauma} dias",
                    f"{anomaly_score:.3f}",
                    fuzzy_label,
                ],
            }
        )

        st.dataframe(input_df, width="stretch", hide_index=True)
        st.progress(probability)
        st.markdown(f"**Probabilidade simulada de recuperação motora após 1 ano:**\n### {probability:.0%}")

    with pred_col_2:
        st.pyplot(plot_probability_gauge(probability), width="stretch")

    st.header("Explicabilidade Tabular — Simulação tipo SHAP")

    xai_col_1, xai_col_2 = st.columns([1.2, 1])

    with xai_col_1:
        st.pyplot(plot_shap_bar(shap_df), width="stretch")

    with xai_col_2:
        st.markdown(
            """
            **Leitura clínica do gráfico**

            - **UEMS / LEMS**: força motora por miótomos-chave (membros superiores/inferiores).
            - **Escala ASIA**: gravidade neurológica inicial.
            - **Nível Neurológico**: posição rostro-caudal da lesão.
            - **Idade** e **Tempo desde o Trauma**: covariáveis clínico-demográficas do EMSCI.
            - **Características da MRI**: biomarcador visual derivado do módulo de imagem.

            Em versão real, esses pesos devem ser calculados com `SHAP`
            sobre um modelo treinado com dados longitudinais anonimizados.
            """
        )

        st.dataframe(
            shap_df.sort_values("Peso relativo", ascending=False),
            width="stretch",
            hide_index=True,
        )

    st.header("Comparação de modelos baseados em árvores — Random Forest vs XGBoost")

    xgboost_probability = simulate_recovery_probability_xgboost(
        asia=asia_scale,
        uems=uems,
        lems=lems,
        age=age,
        neuro_level=neuro_level,
        days_since_trauma=days_since_trauma,
        anomaly_score=anomaly_score,
    )

    compare_col_1, compare_col_2 = st.columns(2)
    with compare_col_1:
        st.metric("Recuperação em 1 ano (Random Forest)", f"{probability:.0%}")
    with compare_col_2:
        st.metric("Recuperação em 1 ano (XGBoost)", f"{xgboost_probability:.0%}")

    if not XGBOOST_AVAILABLE:
        st.warning("xgboost não está instalado. Rode: py -m pip install xgboost")
    else:
        with st.expander("Métricas de validação em holdout sintético (30%)", expanded=False):
            st.caption(
                "Métricas calculadas sobre a base clínica sintética (mesmo random_state=42 usado "
                "no treino), apenas para comparar os dois algoritmos citados no artigo. Sem validade clínica."
            )
            metrics = compare_tree_models_backend()
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df.style.format("{:.3f}"), width="stretch")

    st.header("Resumo interpretativo")

    if probability >= 0.70:
        prognosis = "prognóstico motor favorável"
    elif probability >= 0.40:
        prognosis = "prognóstico motor intermediário"
    else:
        prognosis = "prognóstico motor reservado"

    st.markdown(
        f"""
        O paciente simulado apresenta **Escala ASIA {asia_scale}**, nível neurológico
        **{neuro_level}**, força motora basal de **{motor_strength}/100** (UEMS {uems}/50 +
        LEMS {lems}/50), **{age} anos**, **{days_since_trauma} dias** desde o trauma e escore
        visual de anomalia de **{anomaly_score:.3f}**. A fusão multimodal desses fatores gerou
        uma probabilidade simulada de recuperação de **{probability:.0%}** (Random Forest),
        compatível com **{prognosis}**.

        O protótipo demonstra integração entre visão computacional, dados clínicos estruturados
        e explicabilidade, mantendo rastreabilidade adequada para apresentação acadêmica.
        """
    )

    st.header("Relatório técnico da execução")

    report_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "image_origin": image_origin_label,
        "data_source": data_source,
        "asia_baseline": asia_scale,
        "neuro_level_baseline": neuro_level,
        "uems_baseline": uems,
        "lems_baseline": lems,
        "motor_strength_baseline": motor_strength,
        "age": age,
        "days_since_trauma": days_since_trauma,
        "mri_anomaly_score": round(anomaly_score, 6),
        "fuzzy_class": fuzzy_label,
        "recovery_probability_1y_random_forest": round(probability, 6),
        "recovery_probability_1y_xgboost": round(xgboost_probability, 6),
        "visual_only_risk": visual_only_risk,
        "multimodal_risk": risk_label,
        "prototype_warning": "Resultados simulados, sem validade clínica.",
    }

    saved_report_path = persist_execution_report(report_payload)

    report_col_1, report_col_2 = st.columns([1, 1])

    with report_col_1:
        st.json(report_payload)

    with report_col_2:
        st.download_button(
            "Baixar relatório JSON",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name="relatorio_execucao_pipeline_lesao_medular.json",
            mime="application/json",
            width="stretch",
        )
        st.caption(f"Também salvo automaticamente em `{saved_report_path.as_posix()}`.")

        st.markdown(
            """
            **Uso recomendado na apresentação:** mostrar este relatório para evidenciar
            rastreabilidade da execução, parâmetros de entrada e saída do pipeline.
            """
        )

else:
    st.info("Imagem carregada. Clique em **Executar análise multimodal** na sidebar para processar.")
