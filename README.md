# Sistema Experimental — Pipeline Multimodal para Prognóstico de Lesão Medular por MRI

> **Aviso acadêmico:** este é um protótipo (PoC) experimental. Ele **não realiza diagnóstico
> nem prognóstico clínico real**. Os módulos de machine learning existem para demonstrar
> arquitetura, integração multimodal e explicabilidade — não possuem validade clínica.

Aplicação [Streamlit](https://streamlit.io/) que integra leitura de imagens de MRI/DICOM,
dados clínicos simulados inspirados no protocolo EMSCI (Escala ASIA, força motora por
miótomos-chave UEMS/LEMS, nível neurológico, idade, tempo desde o trauma), explicabilidade
visual (Grad-CAM), lógica fuzzy, um classificador de anomalia visual (Isolation Forest sobre
embeddings de ResNet50) e um modelo clínico (Random Forest + XGBoost, com SHAP) para gerar
uma probabilidade simulada de recuperação motora e um Risco Multimodal explícito.

## Funcionalidades

- Leitura de imagens em PNG, JPG/JPEG e DICOM (upload manual, dataset local, Kaggle API ou URL).
- Extração de embeddings visuais com **ResNet50** (ImageNet) e geração de mapa **Grad-CAM** real.
- Escore de anomalia visual com **Isolation Forest** treinado sobre os embeddings.
- Classificação fuzzy (Leve / Moderado / Severo) a partir do escore de anomalia.
- Predição clínica com **Random Forest** treinado em dados sintéticos + explicabilidade com **SHAP**.
- Comparação lado a lado com **XGBoost** treinado na mesma base sintética (métricas de acurácia/AUC em holdout).
- Painel de configuração persistente (`.streamlit/app_config.json`) para caminhos de dataset,
  limites de listagem e credenciais do Kaggle.
- Download do relatório técnico da execução em JSON, com persistência automática em `reports/`
  (um arquivo por execução, com timestamp no nome, para auditoria posterior).

Quando uma dependência opcional (TensorFlow, SHAP, OpenCV) não está disponível no ambiente,
o app cai automaticamente em um modo simulado/heurístico equivalente, para nunca quebrar a
demonstração.

## Pré-requisitos

- **Python 3.10 a 3.12** (recomendado) para ter o backend visual real com **ResNet50/Grad-CAM**
  instalado automaticamente. Em **Python 3.13/3.14** o app continua funcionando normalmente
  (fuzzy, Random Forest, XGBoost, SHAP, Isolation Forest, leitura de DICOM), mas usando o
  heatmap visual simulado como substituto do Grad-CAM real, pois o TensorFlow ainda não
  possui build oficial para essas versões do Python.
- pip atualizado.
- (Opcional) Conta no [Kaggle](https://www.kaggle.com/) com API Key, caso queira baixar o
  dataset RSNA diretamente pela interface.

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd lesao-medular-poc

# 2. Crie e ative um ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. Instale as dependências
python -m pip install --upgrade pip
python -m pip install -r requirements_streamlit_v2.txt
```

O arquivo `requirements_streamlit_v2.txt` cobre o pipeline completo (Streamlit, scikit-learn,
XGBoost, SHAP, OpenCV, pydicom, kaggle, TensorFlow, etc.). O TensorFlow é instalado
**automaticamente por padrão em Python 3.10, 3.11 e 3.12** (marcador `python_version < "3.13"`
no requirements). Em **Python 3.13/3.14** o `pip install` pula o TensorFlow — ainda não há
build oficial dele para essas versões — e o app cai automaticamente no heatmap visual
simulado como substituto do Grad-CAM real.

## Executando a aplicação

```bash
streamlit run app.py
# no Windows, alternativamente:
py -m streamlit run app.py
```

O app abre automaticamente no navegador em `http://localhost:8501`.

## Configurando o dataset (RSNA 2024 Lumbar Spine)

O app espera imagens DICOM organizadas como:

```text
data/rsna/train_images/{study_id}/{series_id}/{instance}.dcm
data/rsna/train_series_descriptions.csv
```

Existem três formas de obter os dados, todas configuráveis pela aba **Configurações** do app:

### Opção A — Kaggle API pela própria interface

1. Gere sua API Key em [kaggle.com/settings](https://www.kaggle.com/settings) (seção *API* → *Create New Token*), o que baixa um `kaggle.json` com `username` e `key`.
2. Abra o app, vá em **Configurações → Credenciais Kaggle** e informe usuário e chave. Isso salva localmente em `.streamlit/secrets.toml` (arquivo **não deve ser versionado** — veja a seção de segurança abaixo).
3. Ainda na aba **Análise**, selecione a origem **"Kaggle API / baixar dataset"** e use os botões **Baixar** e **Descompactar**.

### Opção B — Kaggle CLI manual

```bash
pip install kaggle
mkdir -p data/rsna
kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification -p data/rsna
python -m zipfile -e data/rsna/rsna-2024-lumbar-spine-degenerative-classification.zip data/rsna
```

### Opção C — Upload manual ou URL

Não é necessário baixar o dataset completo: na aba **Análise**, escolha **"Upload manual"**
para subir uma imagem PNG/JPG/DICOM local, ou **"URL/API de imagem"** para apontar para um
endpoint que retorne os bytes da imagem.

## Configuração persistente do app

A aba **Configurações** permite ajustar e salvar (em `.streamlit/app_config.json`):

| Configuração | Padrão |
|---|---|
| Dataset root | `data/rsna` |
| Pasta de imagens | `data/rsna/train_images` |
| CSV de metadados | `data/rsna/train_series_descriptions.csv` |
| Competição Kaggle | `rsna-2024-lumbar-spine-degenerative-classification` |
| Limite padrão de imagens listadas | `500` |
| Executar análise automaticamente ao carregar imagem | ativado |

Depois de salvar, reinicie o Streamlit para garantir que todas as configurações sejam
aplicadas.

## Segurança: arquivos que não devem ir para o Git

Ao salvar credenciais Kaggle pela interface, o app cria `.streamlit/secrets.toml` com suas
credenciais em texto puro. **Nunca versione esse arquivo.** Um `.gitignore` já está incluído
neste repositório cobrindo:

- `.streamlit/secrets.toml`
- `.streamlit/app_config.json`
- `data/` (o dataset é baixado localmente, não deve ir para o repositório)
- ambientes virtuais, caches e artefatos (`.venv/`, `__pycache__/`, etc.)

## Estrutura do projeto

```text
app.py                        # aplicação Streamlit (ponto de entrada único)
requirements_streamlit_v2.txt # dependências do pipeline
.streamlit/                   # config local e credenciais (gerado em tempo de execução, não versionado)
data/                         # dataset local (não versionado)
reports/                      # relatórios JSON gerados a cada execução (não versionado)
```

> **Nota:** o pacote `src/` presente no repositório contém uma versão simplificada/antiga de
> alguns componentes visuais e não é importado por `app.py`, que concentra toda a lógica real
> do pipeline. Ele é mantido por enquanto por compatibilidade histórica, mas não reflete o
> estado atual da aplicação.

## Maturidade científica do protótipo

| Módulo | Status |
|---|---|
| Leitura real de DICOM | ✅ Implementado |
| Interface clínica demonstrativa | ✅ Implementado |
| Embeddings ResNet50 + Grad-CAM | ✅ Real (TensorFlow instalado por padrão em Python <3.13); fallback simulado em Python 3.13/3.14 |
| Isolation Forest | ✅ Real (requer scikit-learn + embeddings) |
| Lógica fuzzy | ✅ Implementado (heurística) |
| Predição clínica (Random Forest) | ✅ Real, treinado em dados **sintéticos** |
| Predição clínica (XGBoost, comparação) | ✅ Real (requer pacote `xgboost`), treinado em dados **sintéticos** |
| SHAP | ✅ Real (requer pacote `shap`); fallback heurístico quando indisponível |

**Limitação importante:** o dataset usado (RSNA 2024 Lumbar Spine Degenerative
Classification) serve para validar tecnicamente o pipeline com imagens DICOM reais, mas
**não é uma base clínica de lesão medular traumática**, e o modelo clínico é treinado em
dados sintéticos. Resultados exibidos pelo app são simulados e não possuem validade
diagnóstica ou prognóstica.

## Licença

Defina a licença do projeto antes de publicar (ex.: MIT, Apache-2.0). Nenhuma licença está
definida por padrão neste repositório.
