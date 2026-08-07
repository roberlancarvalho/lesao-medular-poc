"""
Smoke tests do pipeline multimodal.

Carrega app.py como módulo (fora do runtime real do Streamlit, que apenas
emite avisos "missing ScriptRunContext" em vez de falhar) e exercita as
funções centrais de cada estágio: fuzzy, Random Forest, XGBoost e o
combinador de Risco Multimodal. Não testam a interface visual — servem
para pegar erros de import/runtime a cada push/PR, antes de chegar no
navegador.
"""

import importlib.util
import warnings
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def app_module():
    warnings.filterwarnings("ignore")
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_module_imports_without_exceptions(app_module):
    assert app_module.APP_VERSION


def test_fuzzy_anomaly_labels(app_module):
    assert app_module.fuzzy_anomaly(0.05)["label"] == "Leve"
    assert app_module.fuzzy_anomaly(0.50)["label"] == "Moderado"
    assert app_module.fuzzy_anomaly(0.95)["label"] == "Severo"


def test_random_forest_probability_is_bounded(app_module):
    probability = app_module.simulate_recovery_probability(
        asia="C", uems=25, lems=25, age=40, neuro_level="C6",
        days_since_trauma=30, anomaly_score=0.2,
    )
    assert 0.0 < probability < 1.0


def test_xgboost_probability_is_bounded(app_module):
    probability = app_module.simulate_recovery_probability_xgboost(
        asia="C", uems=25, lems=25, age=40, neuro_level="C6",
        days_since_trauma=30, anomaly_score=0.2,
    )
    assert 0.0 < probability < 1.0


def test_multimodal_risk_scenario_from_article(app_module):
    """MRI 'Leve' + quadro clínico grave deve resultar em Risco Multimodal Alto (Seção 3.3)."""
    probability = app_module.simulate_recovery_probability(
        asia="A", uems=2, lems=1, age=55, neuro_level="C5",
        days_since_trauma=20, anomaly_score=0.10,
    )
    fuzzy_label = app_module.fuzzy_anomaly(0.10)["label"]
    visual_risk = app_module.classify_visual_only_risk(fuzzy_label)
    multimodal_risk = app_module.classify_multimodal_risk(probability)

    assert fuzzy_label == "Leve"
    assert visual_risk == "Baixo"
    assert multimodal_risk == "Alto"
    assert visual_risk != multimodal_risk


def test_report_persistence(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "REPORTS_DIR", tmp_path)
    payload = {"teste": True}
    saved_path = app_module.persist_execution_report(payload)
    assert saved_path.exists()
    assert saved_path.parent == tmp_path
    assert payload["embedding_path"] is None


def test_report_persistence_with_embedding(app_module, tmp_path, monkeypatch):
    """O embedding real da ResNet50 deve ser salvo à parte e referenciado no relatório."""
    import numpy as np

    monkeypatch.setattr(app_module, "REPORTS_DIR", tmp_path)
    payload = {"teste": True}
    fake_embedding = np.zeros(2048, dtype=np.float32)

    saved_path = app_module.persist_execution_report(payload, embedding=fake_embedding)

    assert saved_path.exists()
    assert payload["embedding_dim"] == 2048
    embedding_path = tmp_path / "embeddings" / Path(payload["embedding_path"]).name
    assert embedding_path.exists()


def test_synthetic_dataset_includes_fuzzy_category(app_module):
    """A categoria fuzzy deve entrar como atributo do modelo, não só o escore contínuo (Seção 3.3)."""
    X, _ = app_module._build_synthetic_clinical_dataset()
    assert "Categoria Fuzzy (ordinal)" in X.columns


def test_patient_features_include_fuzzy_category(app_module):
    patient_X = app_module.build_patient_features(
        asia="A", uems=2, lems=1, age=55, neuro_level="C5",
        days_since_trauma=20, anomaly_score=0.10,
    )
    assert patient_X["Categoria Fuzzy (ordinal)"].iloc[0] == 0.0  # "Leve"


def test_xgboost_shap_importance_runs(app_module):
    shap_df = app_module.simulate_shap_importance_xgboost(
        asia="C", uems=25, lems=25, age=40, neuro_level="C6",
        days_since_trauma=30, anomaly_score=0.2,
    )
    assert "Categoria Fuzzy (ordinal)" in shap_df["Variável"].values
