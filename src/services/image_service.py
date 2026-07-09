from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.math_utils import normalize

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    PYDICOM_AVAILABLE = True
except Exception:
    PYDICOM_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def reset_file_pointer(source) -> None:
    """Tenta voltar o ponteiro do arquivo para o início."""
    try:
        source.seek(0)
    except Exception:
        pass


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

    reset_file_pointer(dicom_source)
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

    reset_file_pointer(dicom_source)
    return Image.fromarray(arr).convert("RGB")


def read_dicom_metadata(dicom_source) -> dict:
    """Lê metadados DICOM sem carregar pixels, quando possível."""
    if not PYDICOM_AVAILABLE:
        return {"Erro": "pydicom não instalado"}

    try:
        reset_file_pointer(dicom_source)
        ds = pydicom.dcmread(dicom_source, stop_before_pixels=True)
        reset_file_pointer(dicom_source)

        metadata = {
            "Modality": str(getattr(ds, "Modality", "N/A")),
            "SeriesDescription": str(getattr(ds, "SeriesDescription", "N/A")),
            "Manufacturer": str(getattr(ds, "Manufacturer", "N/A")),
            "MagneticFieldStrength": str(getattr(ds, "MagneticFieldStrength", "N/A")),
            "SliceThickness": str(getattr(ds, "SliceThickness", "N/A")),
            "Rows": str(getattr(ds, "Rows", "N/A")),
            "Columns": str(getattr(ds, "Columns", "N/A")),
            "PhotometricInterpretation": str(getattr(ds, "PhotometricInterpretation", "N/A")),
        }

        return metadata

    except Exception as exc:
        return {"Erro": str(exc)}


def load_image_any_format(source, filename: str = "") -> Image.Image:
    name = filename.lower()

    if name.endswith(".dcm"):
        return dicom_to_pil(source)

    reset_file_pointer(source)
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


def list_dataset_images(dataset_dir: str, limit: int = 500) -> list[str]:
    root = Path(dataset_dir)

    if not root.exists():
        return []

    files = []

    for ext in ["*.dcm", "*.png", "*.jpg", "*.jpeg"]:
        files.extend(root.rglob(ext))

    files = sorted(files)

    return [str(path) for path in files[:limit]]


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