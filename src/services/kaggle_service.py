import os
import subprocess
import sys
import zipfile
from pathlib import Path


def get_kaggle_credentials(st_secrets=None) -> tuple[str | None, str | None]:
    """
    Busca credenciais Kaggle.

    Ordem:
    1. st.secrets, quando disponível;
    2. variáveis de ambiente.
    """
    username = None
    key = None

    if st_secrets is not None:
        try:
            username = st_secrets.get("KAGGLE_USERNAME")
            key = st_secrets.get("KAGGLE_KEY")
        except Exception:
            pass

    username = username or os.environ.get("KAGGLE_USERNAME")
    key = key or os.environ.get("KAGGLE_KEY")

    return username, key


def check_kaggle_ready(st_secrets=None) -> tuple[bool, str]:
    username, key = get_kaggle_credentials(st_secrets)

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


def download_kaggle_dataset(
    competition: str,
    dataset_root: Path,
    st_secrets=None,
) -> tuple[bool, str]:
    ready, message = check_kaggle_ready(st_secrets)

    if not ready:
        return False, message

    username, key = get_kaggle_credentials(st_secrets)
    dataset_root.mkdir(parents=True, exist_ok=True)

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
        competition,
        "-p",
        str(dataset_root),
    ]

    result = subprocess.run(command, capture_output=True, text=True, env=env)

    output = result.stdout or result.stderr or ""

    if result.returncode != 0:
        return False, output

    return True, output or "Dataset baixado com sucesso."


def unzip_kaggle_dataset(zip_path: Path, dataset_root: Path) -> tuple[bool, str]:
    if not zip_path.exists():
        return False, f"Arquivo ZIP não encontrado em: {zip_path}"

    dataset_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_root)

        return True, "Dataset descompactado com sucesso."

    except Exception as exc:
        return False, str(exc)


def dataset_status(dataset_root: Path, images_dir: Path, zip_path: Path) -> dict:
    zip_exists = zip_path.exists()
    images_dir_exists = images_dir.exists()
    n_dicom_images = 0

    if images_dir_exists:
        n_dicom_images = len(list(images_dir.rglob("*.dcm")))

    return {
        "zip_exists": zip_exists,
        "images_dir_exists": images_dir_exists,
        "n_dicom_images": n_dicom_images,
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "zip_path": str(zip_path),
    }