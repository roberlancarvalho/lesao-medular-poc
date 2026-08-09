# Pipeline Multimodal para Prognóstico de Lesão Medular com Inteligência Artificial

### Uma Arquitetura Baseada em Visão Computacional e Dados Clínicos

[![CI](https://github.com/roberlancarvalho/lesao-medular-poc/actions/workflows/ci.yml/badge.svg)](https://github.com/roberlancarvalho/lesao-medular-poc/actions/workflows/ci.yml)

**Demo pública:** [lesao-medular-poc.streamlit.app](https://lesao-medular-poc.streamlit.app/) ·
**Artigo (PDF):** [PDF/pipeline-multimodal-lesao-medular.pdf](PDF/pipeline-multimodal-lesao-medular.pdf) ·
**Apresentação (PPT/PDF):** [PPT/apresentacao-pipeline-multimodal-lesao-medular.pdf](PPT/apresentacao-pipeline-multimodal-lesao-medular.pdf)

## Sumário

- [Autoria](#autoria)
- [Resumo](#resumo)
- [Capturas de tela](#capturas-de-tela)
- [Arquitetura](#arquitetura)
- [Funcionalidades](#funcionalidades)
- [Instalação e execução](#pré-requisitos)
- [Como interpretar os resultados](#como-interpretar-os-resultados)
- [CI/CD](#cicd)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Maturidade científica do protótipo](#maturidade-científica-do-protótipo)
- [Licença](#licença)
- [Como citar este repositório](#como-citar-este-repositório)
- [Uso de Inteligência Artificial Generativa](#uso-de-inteligência-artificial-generativa)
- [Referências](#referências)

## Autoria

- **Autor:** Roberlan Oliveira de Carvalho ([roberlan.carvalho@gmail.com](mailto:roberlan.carvalho@gmail.com))
- **Programa:** Programa de Pós-Graduação em Informática, Universidade Federal do Rio de Janeiro (UFRJ)
- **Orientação:** trabalho sem orientador formal associado.

> **Aviso acadêmico:** este é um protótipo (PoC) experimental. Ele **não realiza diagnóstico
> nem prognóstico clínico real**. Os módulos de machine learning existem para demonstrar
> arquitetura, integração multimodal e explicabilidade — não possuem validade clínica.

## Resumo

Este trabalho propõe uma Prova de Conceito (PoC) de uma arquitetura multimodal de
Inteligência Artificial para apoio ao prognóstico de recuperação motora em lesão medular
traumática. A proposta integra imagens de ressonância magnética (MRI) no padrão *Digital
Imaging and Communications in Medicine* (DICOM) e dados clínicos estruturados, como escala
ASIA e força motora. O pipeline organiza-se em dois módulos: visão computacional, com
extração de representações por ResNet50, detecção de anomalias por Isolation Forest e
interpretação por lógica fuzzy; e predição clínica multimodal, baseada em modelos de árvores,
como Random Forest e XGBoost. A explicabilidade é apoiada por Grad-CAM e SHAP. A PoC combina
imagens DICOM reais com dados clínicos sintéticos, demonstrando viabilidade técnica inicial,
sem validade clínica, dependente de dados longitudinais reais.

**Palavras-chave:** Lesão Medular; Inteligência Artificial Explicável; Aprendizado Multimodal;
Visão Computacional; Lógica Fuzzy.

Este repositório é o protótipo de software (PoC) descrito no artigo acima — o app Streamlit
descrito a seguir implementa a arquitetura de dois módulos, a camada fuzzy e a explicabilidade
(Grad-CAM + SHAP) discutidas no texto completo, disponível em
[`PDF/pipeline-multimodal-lesao-medular.pdf`](PDF/pipeline-multimodal-lesao-medular.pdf).

## Capturas de tela

**Visão geral da execução** — MRI carregada, mapa Grad-CAM, interpretação fuzzy, predição
clínica (Random Forest x XGBoost), explicabilidade SHAP e relatório técnico:

![Visão geral do app: MRI, Grad-CAM, fuzzy, predição clínica, SHAP e relatório JSON](docs/screenshots/visao-geral-app.png)

**Topo da página e desacoplamento Módulo 1 vs Módulo 2** — aviso acadêmico, indicadores
principais (escore de anomalia, fuzzy, probabilidade, risco) e o painel que contrasta o risco
calculado só a partir da imagem (visual isolado) com o Risco Multimodal final, após a fusão
com o quadro clínico:

![Painel de Risco Multimodal comparando o risco visual isolado com o risco após a fusão clínica](docs/screenshots/risco-multimodal-desacoplamento.png)

> No exemplo acima, os dois riscos coincidem (**Baixo**). Quando o quadro clínico é grave (ex.:
> Escala ASIA A, força motora baixa) mesmo com uma MRI classificada como "Leve", o painel
> destaca a divergência com um aviso — evidenciando que o Risco Multimodal pondera o contexto
> clínico em vez de replicar diretamente o sinal da imagem.

## Arquitetura

A arquitetura (detalhada na Seção 3 do artigo) organiza-se em dois módulos integrados, numa
fusão tardia em nível tabular — não uma fusão fim a fim opaca — favorecendo desacoplamento
funcional, reprodutibilidade computacional e rastreabilidade analítica:

```text
┌─────────────────────────────┐      ┌──────────────────────────────────┐
│  Módulo 1 — Visão            │      │  Módulo 2 — Predição Clínica      │
│  Computacional                │      │  Multimodal                       │
│                               │      │                                    │
│  MRI/DICOM (RSNA 2024)       │      │  Escala ASIA · Força Motora        │
│        │                     │      │  (UEMS/LEMS) · Nível Neurológico   │
│        ▼                     │      │  · Idade · Tempo desde o Trauma    │
│  Pré-processamento           │      │        │                           │
│  (leitura DICOM + normaliz.) │      │        ▼                           │
│        │                     │      │  Fusão tabular tardia              │
│        ▼                     │      │  (+ escore MRI e categoria fuzzy)  │
│  ResNet50 pré-treinada       │─────▶│        │                           │
│  (extração de embeddings)    │ escore│        ▼                          │
│        │            ┌────────┘ MRI  │  Random Forest / XGBoost           │
│        ▼            │              │  (modelo preditivo tabular)         │
│  Isolation Forest    │ Grad-CAM     │        │                           │
│  (anomalia visual)   │ (explicab.   │        ▼                           │
│        │             │  visual)     │  Probabilidade de recuperação      │
│        ▼             │              │  motora em 1 ano                   │
│  Lógica Fuzzy         │              │        │                           │
│  (Leve/Moderada/      │              │        ▼                          │
│   Severa)             │              │  Risco Multimodal ─── SHAP         │
└───────────────────────┘              └────────────────────────────────────┘
```

- **Módulo 1** recebe a MRI, extrai atributos visuais por transferência de aprendizado
  (ResNet50), identifica padrões anômalos com Isolation Forest (algoritmo não supervisionado)
  e traduz o escore contínuo em categoria linguística de incerteza via lógica fuzzy (Anomalia
  Leve/Moderada/Severa). O Grad-CAM explica espacialmente onde a rede concentrou ativação.
- **Módulo 2** integra esse escore (+ sua categoria fuzzy) a variáveis clínicas inspiradas no
  protocolo EMSCI e gera a predição de recuperação motora com modelos baseados em árvores
  (Random Forest e XGBoost), explicados por SHAP (TreeSHAP).
- **Desacoplamento funcional**: a classificação fuzzy do Módulo 1 depende só dos pixels da
  imagem; o Módulo 2 pondera esse sinal junto ao quadro clínico numa fusão tardia — por isso
  uma MRI "Leve" pode coexistir com "Alto Risco Multimodal" quando o quadro clínico é grave
  (ver bloco de desacoplamento nas capturas de tela acima e a Seção 3.3 do artigo).

Este README documenta a implementação de software (o app Streamlit); a arquitetura conceitual
completa — incluindo a Figura 1 original, a discussão de proveniência de dados (Seção 3.1.1) e
a análise comparativa com trabalhos relacionados (Seção 2) — está no
[artigo em PDF](PDF/pipeline-multimodal-lesao-medular.pdf).

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

- **Python 3.10 a 3.13** (recomendado) para ter o backend visual real com **ResNet50/Grad-CAM**
  instalado automaticamente. Em **Python 3.14** o app continua funcionando normalmente (fuzzy,
  Random Forest, XGBoost, SHAP, Isolation Forest, leitura de DICOM), mas usando o heatmap
  visual simulado como substituto do Grad-CAM real, pois o TensorFlow ainda não possui build
  oficial para essa versão do Python.
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
python -m pip install -r requirements.txt
```

O arquivo `requirements.txt` cobre o pipeline completo, com **todas as versões fixadas**
(Streamlit, scikit-learn, XGBoost, SHAP, OpenCV, pydicom, kaggle, TensorFlow, etc.), conforme
a Seção 3.1.2 do artigo. O TensorFlow é instalado **automaticamente em Python 3.10 a 3.13**
(marcador `python_version < "3.14"` no requirements). Em **Python 3.14** o `pip install` pula
o TensorFlow — ainda não há build oficial dele para essa versão — e o app cai automaticamente
no heatmap visual simulado como substituto do Grad-CAM real.

## Executando a aplicação

```bash
streamlit run app.py
# no Windows, alternativamente:
py -m streamlit run app.py
```

O app abre automaticamente no navegador em `http://localhost:8501`.

## Como interpretar os resultados

Depois de carregar uma imagem e preencher os dados clínicos na barra lateral, clique em
**"Executar análise multimodal"** (ou deixe rodar automaticamente, se a opção estiver ativa
em Configurações). A tela principal mostra, em sequência:

1. **Escore de Anomalia MRI**, **Classificação Fuzzy** (Leve/Moderado/Severo) e a
   **probabilidade simulada de recuperação** em 1 ano.
2. **Risco visual isolado vs Risco Multimodal** — o ponto mais importante para entender a
   proposta do artigo. O primeiro vem só da imagem (Módulo 1); o segundo vem da fusão com o
   quadro clínico (Módulo 2). Quando eles **divergem**, o app avisa explicitamente — é o caso,
   por exemplo, de uma MRI "Leve" combinada com Escala ASIA A e força motora mínima, que ainda
   assim resulta em Risco Multimodal **Alto**.
3. **Módulo 1 (visão computacional)**: a MRI carregada lado a lado com o mapa Grad-CAM e a
   tabela de graus de pertinência fuzzy.
4. **Módulo 2 (predição clínica)**: as variáveis EMSCI usadas (ASIA, nível neurológico,
   UEMS/LEMS, idade, tempo desde o trauma), o gráfico de probabilidade e a explicabilidade
   SHAP.
5. **Comparação Random Forest vs XGBoost**, com métricas de acurácia/AUC em holdout sintético
   e explicabilidade SHAP para os dois modelos.
6. **Relatório técnico**: JSON com todos os parâmetros e saídas da execução, disponível para
   download e também salvo automaticamente em `reports/`. Quando o backend visual real está
   ativo, o embedding da ResNet50 (2048-d) também é salvo em `reports/embeddings/` e referenciado
   no relatório (`embedding_path`), para auditoria posterior.

Para reproduzir o cenário de desacoplamento descrito no artigo (imagem "leve" não implica
risco baixo), tente **Escala ASIA A**, **UEMS/LEMS próximos de 0** e uma imagem com escore de
anomalia baixo — o Risco Multimodal deve virar **Alto** mesmo com a classificação fuzzy
"Leve".

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

## CI/CD

### CI — verificação automática (GitHub Actions)

A cada `push`/pull request na branch `main`, o workflow em
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) roda em Python 3.11, 3.12, 3.13 e 3.14
(matriz), fazendo:

1. `python -m py_compile app.py` — garante que o arquivo não tem erro de sintaxe.
2. `ruff check --select F app.py` — lint (imports/variáveis não usadas, nomes indefinidos).
3. `pytest tests/` — smoke tests que carregam o `app.py` e exercitam fuzzy, Random Forest,
   XGBoost, o combinador de Risco Multimodal e a persistência do relatório (ver
   [`tests/test_smoke.py`](tests/test_smoke.py)).

A versão 3.14 na matriz também serve para garantir que o **fallback sem TensorFlow** (heatmap
simulado no lugar do Grad-CAM real) continua funcionando, já que o `requirements.txt` pula o
TensorFlow nessa versão do Python.

Para rodar os mesmos testes localmente antes de dar push:

```bash
pip install pytest ruff
python -m py_compile app.py
ruff check --select F app.py
pytest tests/ -v
```

### CD — deploy gratuito no Streamlit Community Cloud

O app pode ser publicado de graça em [share.streamlit.io](https://share.streamlit.io/) (conta
GitHub). O deploy lá é automático a partir do repositório conectado — não depende de um
workflow de CD no GitHub Actions, só da configuração inicial abaixo. Este projeto já está
publicado em **[lesao-medular-poc.streamlit.app](https://lesao-medular-poc.streamlit.app/)**,
reimplantado automaticamente a cada push na `main`.

1. Acesse [share.streamlit.io](https://share.streamlit.io/), entre com sua conta GitHub e
   clique em **"New app"**.
2. Selecione o repositório `lesao-medular-poc`, a branch `main` e o arquivo principal `app.py`.
3. Em **"Advanced settings"**, defina a versão do Python como **3.11, 3.12 ou 3.13** (o que
   estiver disponível no seletor do Streamlit Cloud) — é o que faz o `requirements.txt`
   instalar o TensorFlow de verdade (backend visual real do Grad-CAM). Se o seletor só oferecer
   uma versão mais nova sem TensorFlow disponível ainda, o deploy funciona do mesmo jeito, só
   cai no fallback simulado.
4. Clique em **"Deploy"**. Qualquer novo `push` na `main` reimplanta automaticamente.

**Duas observações importantes para essa PoC especificamente:**

- **Modo de dados recomendado para demo pública:** use **"Upload manual"** na aba Análise — não
  depende de dataset nem de credenciais Kaggle, funciona direto no ar. O modo "Kaggle API /
  baixar dataset" baixa vários GB e pode não caber no disco/tempo do plano gratuito.
- **Credenciais Kaggle (opcional):** se quiser habilitar o download do dataset no app já
  publicado, não use a tela de "Configurações" do próprio app em produção (ela grava em
  `.streamlit/secrets.toml`, que não persiste entre reinicializações no Cloud). Em vez disso,
  configure `KAGGLE_USERNAME` e `KAGGLE_KEY` no menu **"Secrets"** do painel do Streamlit
  Community Cloud — o app já lê essas credenciais de variável de ambiente/`st.secrets`
  automaticamente (`get_kaggle_credentials`).

## Estrutura do projeto

```text
app.py                        # aplicação Streamlit (ponto de entrada único, concentra todo o pipeline) — CÓDIGO
requirements.txt              # dependências do pipeline (nome exigido pelo Streamlit Community Cloud)
tests/                        # smoke tests automatizados (pytest) — CÓDIGO
.github/workflows/ci.yml      # CI: lint + smoke test a cada push/PR
PDF/                          # artigo completo em PDF
PPT/                          # apresentação (slides) em PDF
docs/screenshots/             # imagens/capturas de tela usadas no README — IMAGENS
.streamlit/                   # config local e credenciais (gerado em tempo de execução, não versionado)
data/                         # dataset local (RSNA, baixado via Kaggle) — DATA (não versionado)
reports/                      # relatórios JSON + reports/embeddings/ (não versionado)
```

> **Nota sobre nomenclatura:** este repositório usa `app.py`/`tests/` como código (em vez de
> uma pasta `CODIGO/`) porque o Streamlit Community Cloud e o GitHub Actions exigem esses
> arquivos em locais específicos da raiz do projeto para funcionar (deploy automático e CI).
> Da mesma forma, `data/` não é versionado por conter apenas um dataset público de terceiros
> (RSNA/Kaggle, baixado sob demanda — ver seção "Configurando o dataset" abaixo) e nenhum dado
> foi coletado como parte deste trabalho.

## Maturidade científica do protótipo

| Módulo | Status |
|---|---|
| Leitura real de DICOM | ✅ Implementado |
| Interface clínica demonstrativa | ✅ Implementado |
| Embeddings ResNet50 + Grad-CAM | ✅ Real (TensorFlow instalado por padrão em Python 3.10–3.13); fallback simulado em Python 3.14 |
| Embeddings auditáveis | ✅ Persistidos em `reports/embeddings/` a cada execução real, referenciados no relatório JSON |
| Isolation Forest | ✅ Real (requer scikit-learn + embeddings) |
| Lógica fuzzy | ✅ Implementado (heurística); categoria fuzzy também entra como atributo do modelo tabular (não só o escore contínuo) |
| Predição clínica (Random Forest) | ✅ Real, treinado em dados **sintéticos** |
| Predição clínica (XGBoost, comparação) | ✅ Real (requer pacote `xgboost`), treinado em dados **sintéticos** |
| SHAP | ✅ Real sobre Random Forest **e** XGBoost (requer pacote `shap`); fallback heurístico quando indisponível |
| Dependências fixadas | ✅ `requirements.txt` com versões exatas (`==`), testadas em conjunto |

**Limitação importante:** o dataset usado (RSNA 2024 Lumbar Spine Degenerative
Classification) serve para validar tecnicamente o pipeline com imagens DICOM reais, mas
**não é uma base clínica de lesão medular traumática**, e o modelo clínico é treinado em
dados sintéticos. Resultados exibidos pelo app são simulados e não possuem validade
diagnóstica ou prognóstica.

## Licença

Este repositório está licenciado sob a **Licença MIT** — veja o arquivo
[`LICENSE`](LICENSE) para o texto completo. Em resumo: uso, cópia, modificação e distribuição
são livres, inclusive para fins comerciais, desde que o aviso de copyright original seja
mantido. O código é fornecido "como está", sem garantias — o que é especialmente relevante
dado o aviso acadêmico no topo deste README (resultados simulados, sem validade clínica).

## Como citar este repositório

Se este repositório ou o artigo associado forem úteis para seu trabalho, a referência
sugerida (formato ABNT, mesmo usado pelo próprio artigo para autocitação) é:

```
CARVALHO, R. O. lesao-medular-poc: repositório do protótipo experimental do Pipeline
Multimodal para Prognóstico de Lesão Medular. GitHub, 2026. Disponível em:
https://github.com/roberlancarvalho/lesao-medular-poc. Acesso em: [data de acesso].
```

Para citar o artigo em si:

```
CARVALHO, R. O. Pipeline Multimodal para Prognóstico de Lesão Medular com Inteligência
Artificial: Uma Arquitetura Baseada em Visão Computacional e Dados Clínicos. Programa de
Pós-Graduação em Informática, Universidade Federal do Rio de Janeiro (UFRJ), 2026.
```

BibTeX:

```bibtex
@misc{carvalho2026lesaomedularpoc,
  author = {Carvalho, Roberlan Oliveira de},
  title  = {Pipeline Multimodal para Prognóstico de Lesão Medular com Inteligência
            Artificial: Uma Arquitetura Baseada em Visão Computacional e Dados Clínicos},
  year   = {2026},
  howpublished = {\url{https://github.com/roberlancarvalho/lesao-medular-poc}},
  note   = {Programa de Pós-Graduação em Informática, UFRJ}
}
```

## Uso de Inteligência Artificial Generativa

Conforme declarado na Seção "Conclusão e Trabalhos Futuros" do
[artigo](PDF/pipeline-multimodal-lesao-medular.pdf):

> "Ferramentas de IA generativa, especificamente o OpenAI Codex e o Claude Code (Anthropic),
> foram utilizadas como apoio colaborativo no processo de engenharia de software, incluindo a
> implementação da interface gráfica e a orquestração de código. Todas as decisões
> metodológicas, curadoria bibliográfica, validação técnica e revisão final permaneceram sob
> responsabilidade do autor."

Detalhando o que isso significou na prática, neste repositório especificamente:

- **Implementação de código:** grande parte do `app.py` (módulos de visão computacional,
  predição clínica, explicabilidade, persistência de relatórios/embeddings, testes em
  `tests/test_smoke.py` e o workflow de CI em `.github/workflows/ci.yml`) foi escrita com
  apoio do **Claude Code** (Anthropic), em pares com o autor — incluindo depuração de
  incompatibilidades de dependências e correção de erros de sintaxe/CI.
- **Interface gráfica:** a interface Streamlit (layout, componentes de carregamento
  `st.status`/`st.skeleton`, textos de interpretação exibidos ao usuário) foi implementada com
  apoio de IA generativa e revisada pelo autor.
- **Documentação:** este README (estrutura, texto explicativo) foi organizado com apoio de IA
  generativa a partir do conteúdo do artigo e das decisões tomadas pelo autor.

**Nota:** a concepção conceitual da pesquisa (o problema a resolver, a escolha da arquitetura
em dois módulos, a camada fuzzy como diferencial metodológico, a curadoria bibliográfica e a
interpretação crítica das limitações científicas do protótipo — Tabela 2 do artigo) foi de
autoria inteiramente humana. A IA generativa funcionou como ferramenta de engenharia de
software sobre decisões já definidas pelo autor, não como autora do conteúdo científico.

## Referências

Bibliografia citada no artigo associado a este repositório:

- CAI, L.; BAI, R.; CAO, Q.; SUN, W.; WANG, F.; LIU, X.; LIANG, B.; JIANG, M.; WANG, G.; SHAO,
  Q.; JIANG, X.; WANG, C.; CHEN, C.; TAN, Z.; WU, Q.; BAO, M.; YU, H.; LI, P.; YANG, X.; LU, Q.
  A non-invasive MRI-based multimodal fusion deep learning model (MF-DLM) for predicting
  overall survival in bladder cancer: a multicentre retrospective study. *eClinicalMedicine*,
  v. 90, art. 103640, 2025. Disponível em:
  [linkinghub.elsevier.com/retrieve/pii/S2589537025005747](https://linkinghub.elsevier.com/retrieve/pii/S2589537025005747).
- CARVALHO, R. O. lesao-medular-poc: repositório do protótipo experimental do Pipeline
  Multimodal para Prognóstico de Lesão Medular. GitHub, 2026. Disponível em:
  [github.com/roberlancarvalho/lesao-medular-poc](https://github.com/roberlancarvalho/lesao-medular-poc).
- DEVI, S. R.; JUSTUS, J. J.; VANATHI, M.; VEERAMALLU, B.; ARUNA, V.; MANJUNATH, T. C.;
  SAPAYEV, V. O. U.; BANU, S. An Explainable Machine Learning Model for Early Detection of
  Brain Tumors: Integrating Multi-Modal Medical Imaging and Intelligent Feature Fusion.
  *Engineering, Technology & Applied Science Research*, v. 15, n. 5, p. 26448-26453, 2025.
  Disponível em:
  [etasr.com/index.php/ETASR/article/view/11539](https://etasr.com/index.php/ETASR/article/view/11539).
- EMSCI — EUROPEAN MULTICENTER STUDY ABOUT SPINAL CORD INJURY. About EMSCI. 2024. Disponível
  em: [emsci.org](https://www.emsci.org/).
- GILL, S. S.; PONNIAH, H. S.; GIERSZTEIN, S.; ANANTHARAJ, R. M.; NAMIREDDY, S.; KILLILEA, J.;
  RAMSAY, D. S. C.; SALIH, A.; THAVARAJASINGAM, A.; SCURTU, D.; JANKOVIC, D.; RUSSO, S.;
  KRAMER, A.; THAVARAJASINGAM, S. G. The diagnostic and prognostic capability of artificial
  intelligence in spinal cord injury: A systematic review. *Brain and Spine*, v. 5, art.
  104208, 2025. Disponível em:
  [pmc.ncbi.nlm.nih.gov/articles/PMC11871462](https://pmc.ncbi.nlm.nih.gov/articles/PMC11871462/).
- KRONES, F.; MARIKKAR, U.; PARSONS, G.; SZMUL, A.; MAHDI, A. Review of multimodal machine
  learning approaches in healthcare. *Information Fusion*, v. 114, art. 102690, 2025.
  Disponível em: [doi.org/10.1016/j.inffus.2024.102690](https://doi.org/10.1016/j.inffus.2024.102690).
- LIN, F.; WANG, K.; LAI, M.; WU, Y.; CHEN, C.; WANG, Y.; WANG, R. Multicenter study on
  predicting postoperative upper limb muscle strength improvement in cervical spinal cord
  injury patients using radiomics and deep learning. *Scientific Reports*, v. 15, art. 5805,
  2025. Disponível em:
  [nature.com/articles/s41598-024-72539-0](https://www.nature.com/articles/s41598-024-72539-0).
- MAKI, S.; FURUYA, T.; INOUE, M.; SHIGA, Y.; INAGE, K.; EGUCHI, Y.; ORITA, S.; OHTORI, S.
  Machine Learning and Deep Learning in Spinal Injury: A Narrative Review of Algorithms in
  Diagnosis and Prognosis. *Journal of Clinical Medicine*, v. 13, n. 3, p. 705, 2024.
  Disponível em: [mdpi.com/2077-0383/13/3/705](https://www.mdpi.com/2077-0383/13/3/705).
- RSNA — RADIOLOGICAL SOCIETY OF NORTH AMERICA. RSNA 2024 Lumbar Spine Degenerative
  Classification. Kaggle, 2024. Disponível em:
  [kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification).
- SHIMIZU, T.; INOMATA, K.; SUDA, K.; HARMON, S. M.; KOMATSU, M.; OTA, M.; USHIROZAKO, H.;
  MINAMI, A.; MAKI, S.; ENDO, T.; YAMADA, K.; IWASAKI, N.; TAKAHASHI, H.; YAMAZAKI, M.; KODA,
  M. A multimodal machine learning model integrating clinical and MRI data for predicting
  neurological outcomes following surgical treatment for cervical spinal cord injury.
  *European Spine Journal*, v. 34, n. 9, p. 3747-3755, 2025. Disponível em:
  [link.springer.com/10.1007/s00586-025-08873-2](https://link.springer.com/10.1007/s00586-025-08873-2).
- TORRES, A.; NIETO, J. J. Fuzzy Logic in Medicine and Bioinformatics. *Journal of Biomedicine
  and Biotechnology*, v. 2006, art. ID 91908, 2006. Disponível em:
  [pmc.ncbi.nlm.nih.gov/articles/PMC1559939](https://pmc.ncbi.nlm.nih.gov/articles/PMC1559939/).
- WARNER, E.; LEE, J.; HSU, W.; SYEDA-MAHMOOD, T.; KAHN JR., C. E.; GEVAERT, O.; RAO, A.
  Multimodal Machine Learning in Image-Based and Clinical Biomedicine: Survey and Prospects.
  *International Journal of Computer Vision*, v. 132, n. 9, p. 3753-3769, 2024. Disponível em:
  [link.springer.com/10.1007/s11263-024-02032-8](https://link.springer.com/10.1007/s11263-024-02032-8).
- ZAITSEVA, E.; RABCAN, J.; KVASSAY, M.; LEVASHENKO, V. A New Fuzzy-Based Classification
  Method for Use in Smart/Precision Medicine. *Bioengineering*, v. 10, n. 7, art. 838, 2023.
  Disponível em:
  [pmc.ncbi.nlm.nih.gov/articles/PMC10376790](https://pmc.ncbi.nlm.nih.gov/articles/PMC10376790/).
- ZHANG, H.; YANG, Y.-F.; SONG, X.-L.; HU, H.-J.; YANG, Y.-Y.; ZHU, X.; YANG, C. An
  interpretable artificial intelligence model based on CT for prognosis of intracerebral
  hemorrhage: a multicenter study. *BMC Medical Imaging*, v. 24, n. 1, art. 170, 2024.
  Disponível em:
  [pmc.ncbi.nlm.nih.gov/articles/PMC11234657](https://pmc.ncbi.nlm.nih.gov/articles/PMC11234657/).
