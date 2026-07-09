import json
import os
from pathlib import Path


APP_VERSION = "v3.4-modular"
DEFAULT_KAGGLE_COMPETITION = "rsna-2024-lumbar-spine-degenerative-classification"

CONFIG_DIR = Path(".streamlit")
APP_CONFIG_PATH = CONFIG_DIR / "app_config.json"
SECRETS_PATH = CONFIG_DIR / "secrets.toml"

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


def save_kaggle_secrets(username: str, key: str) -> None:
    """
    Salva credenciais do Kaggle em .streamlit/secrets.toml.

    Atenção:
    nunca versionar esse arquivo no Git.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    username = username.strip()
    key = key.strip()

    content = (
        f'KAGGLE_USERNAME = "{username}"\n'
        f'KAGGLE_KEY = "{key}"\n'
    )

    with SECRETS_PATH.open("w", encoding="utf-8") as file:
        file.write(content)

    # Disponibiliza na sessão atual sem precisar reiniciar imediatamente.
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key