# Python library notebooks

Twenty-eight practical notebooks covering numerical computing, tables, visualization,
statistics, machine learning, NLP, experiment tracking, vector search, images, HTTP,
databases, and a small prediction API.
Run cells from top to bottom in a fresh kernel. Examples generate their own data,
use bundled library datasets, or download the explicitly documented datasets below.

## Setup

Open a terminal in this `Libraries` directory. Use a separate virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name libraries --display-name "Python (Libraries)"
```

Open notebooks in your notebook editor and select **Python (Libraries)**. Start
them with this directory as the working directory so imports and relative paths
resolve correctly. On Windows, activate with `.venv\Scripts\activate`.
The direct dependency pins were resolved for Python 3.9 on macOS Apple Silicon;
other Python versions and platforms have not been verified. The requirements include
both deep-learning frameworks, which are large downloads. OpenCV uses the headless
package because the notebook displays images with Matplotlib rather than GUI windows.

## Suggested learning order

| Stage | Notebooks | Main topics |
| --- | --- | --- |
| 1. Arrays and tables | NumPy, Pandas | Shapes, axes, broadcasting, cleaning, joins, windows, reports |
| 2. Visualization | Matplotlib, Seaborn, Plotly | Static charts, statistical plots, interactive HTML |
| 3. Scientific computing | SciPy, SymPy, Statsmodels | Numerical methods, symbolic mathematics, statistical inference |
| 4. Data systems | Polars, PyArrow, DuckDB, SQLAlchemy | Expressions, lazy queries, typed columns, Parquet, relational data |
| 5. Web data | Requests, BeautifulSoup | HTTP behavior, pagination, HTML parsing, structured exports |
| 6. Images | Pillow, OpenCV | Image transformations, masks, segmentation, object measurements |
| 7. Machine learning | Scikit-learn, XGBoost | Splits, pipelines, baselines, tuning, evaluation, persistence |
| 8. Neural networks | PyTorch, TensorFlow | Tensors, gradients, training, validation, digit recognition |
| 9. Serving and contracts | FastAPI, Pydantic | Validated requests, nested schemas, input audits, predictions, API tests |
| 10. NLP | Transformers, SentenceTransformers | Tokenization, classification, pretrained embeddings, semantic retrieval and evaluation |
| 11. Experiments | Optuna, MLflow | Search budgets, pruning, tracking, model signatures and registry |
| 12. Retrieval | FAISS, Qdrant | Vector metrics, relevance checks, approximate search and payload filters |

The notebooks are independent except for their shared local dataset helper.
FastAPI trains its own model; it does not require finishing the neural-network notebooks.

## Datasets and offline use

- **Pandas:** `data/sales_sample.csv` is a bundled synthetic teaching dataset.
- **Polars:** UCI Wine Quality red and white CSVs.
- **SciPy:** NOAA Mauna Loa monthly CO₂ CSV. Its contents can change as NOAA updates it.

Download the three remote files once while connected:

```sh
python dataset_cache.py
python dataset_cache.py --offline
```

Polars and SciPy automatically download missing files and otherwise reuse
`data/cache/`. To use another machine offline, copy this cache with the repository.
If a download fails, the error supplies the source URL and exact destination for
a manual download. Cached files are ignored by Git. Delete a specific cached file
only when you intentionally want a fresh download; preserve your cache to reproduce
the same dataset snapshot.

Requests starts a practice HTTP server on the local machine; it requires local
socket access but no internet service. Other datasets are local or generated. SentenceTransformers also needs a one-time pretrained-model download, described below.

## Run the digit API

First run `FastAPI.ipynb` from top to bottom. It trains a random forest, writes
`outputs/digits_random_forest.joblib`, generates `digit_api.py`, and exercises the API
using `TestClient`. Retrain after changing the scikit-learn environment.

```sh
python -m uvicorn digit_api:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API documentation.
`GET /health` checks readiness; `GET /model` describes the model.
`POST /predict` accepts `{"image": [...]}` with exactly eight rows of eight integer
pixels in the range 0–16. `POST /predict/batch` accepts an `images` list containing
1–32 objects of that same shape. These are 8×8 digit inputs, not arbitrary uploaded images.

## Verify notebooks

```sh
python scripts/validate_notebooks.py
# Or select individual notebooks:
python scripts/validate_notebooks.py NumPy.ipynb Pandas.ipynb
```

The validator uses the invoking Python environment and a fresh kernel and temporary
working directory for each notebook. It copies the local data, executes every cell,
and exits nonzero if any notebook fails. The default timeout is 600 seconds per cell;
adjust with `--timeout`. Local Jupyter kernel sockets must be allowed.

Executed notebook copies and `report.json` are written under `outputs/validation/`.
The report describes the latest invocation. Source notebooks and existing model
artifacts are not overwritten by validation. Normal interactive notebook runs may
write plots, reports, models, and (for FastAPI) the service module into this directory.

Use the executed copies and `outputs/validation/report.json` to inspect the latest validator invocation.

## DuckDB: learning through an analytics project

[DuckDB.ipynb](DuckDB.ipynb) investigates a
synthetic online store through practical questions, SQL examples, interpretation,
and validation. Topics include cleaning, revenue analysis, customer segmentation,
Parquet, query plans, and time-aware features for predicting repeat purchases.
The notebook generates its dataset locally and writes artifacts under
`outputs/duckdb/`; it works offline once dependencies are installed.

## How the notebooks teach

Each library starts from a practical situation and connects examples to decisions:
what to predict before execution, why an operation fits the question, how to check
the result, and what the evidence does and does not support. Library-specific
checkpoints and closing exercises develop analytical reasoning alongside syntax,
without standalone “Problem:” and “Solution:” labels.

The initial teaching revision covered 21 notebooks; the five additions below follow the same style. The 20 existing notebooks retain
their executable code; their revised notebook structures were validated. The new
DuckDB notebook was also executed end to end: all 19 code cells passed on Python
3.9.6 with DuckDB 1.4.3. Its executed copy is under
`outputs/validation/DuckDB.ipynb` after validation.

## NLP, experiments and retrieval

Each of these notebooks contains its own explanations, dataset, code, analytical
checkpoints, exercises and official references. There are no separate guide files.
They run independently from top to bottom.

| Notebook | Practical investigation |
| --- | --- |
| [Transformers.ipynb](Transformers.ipynb) | Route synthetic support messages; compare tiny BERT training with a TF-IDF baseline, inspect tokens and reload the saved model |
| [Optuna.ipynb](Optuna.ipynb) | Tune a digit classifier with a bounded budget, pruning, persisted studies and an honest held-out comparison |
| [MLflow.ipynb](MLflow.ipynb) | Track digit-model candidates, preserve data lineage, reload artifacts and create a versioned local registry reference |
| [FAISS.ipynb](FAISS.ipynb) | Retrieve fictional help articles, verify vector metrics and compare exact search with IVF/HNSW approximation |
| [Qdrant.ipynb](Qdrant.ipynb) | Search a local knowledge base with tenant/publication filters, payload updates and persistence checks |

All default paths work without internet after dependency installation. Transformers
trains a tiny model from scratch on synthetic text; its optional pretrained-model
download is disabled by default. FAISS and Qdrant use local lexical representations,
not downloaded semantic embeddings. Their small relevance examples are teaching
checks rather than production benchmarks. Qdrant uses persistent local mode and
requires no server. MLflow uses `mlflow-skinny` plus local SQLite/artifacts, which
preserves the existing PyArrow version without installing UI/server dependencies.

Normal runs create model/data artifacts under the corresponding `outputs/`
subdirectory. Optuna studies, MLflow history and Qdrant collections accumulate with
unique identifiers; report exports are replaced. The notebooks explain those
boundaries and how to close or reopen their local resources.

The five additions were executed successfully in fresh kernels on Python 3.9.6:
Transformers (12 code cells), Optuna (11), MLflow (10), FAISS (11), and Qdrant (12).
Their current source code matches the passing executed copies under
`outputs/validation/`. The optional pretrained download and server/UI snippets
were not run. `pip check` reported no broken requirements in the checked environment.

## Semantic embeddings and input contracts

- [SentenceTransformers.ipynb](SentenceTransformers.ipynb) compares pretrained MiniLM
  semantic retrieval with TF-IDF on an authored help-center dataset. It includes
  token-aware chunking, relevance metrics, no-answer threshold selection using only
  development queries, corpus clustering and embedding/encoder reload checks.
- [Pydantic.ipynb](Pydantic.ipynb) validates delivery-estimate requests using strict
  fields, nested models, discriminated unions and cross-field rules. Its project
  reconciles accepted/rejected records, derives features, exports JSON Schema and
  checks serialization. Its final ETA rule is explicitly a fictional teaching
  calculation, not a trained logistics model.

Both contain all explanations and exercises inside their notebooks. No supporting
Python tutorial file or separate guide is required. Pydantic runs offline after
setup. SentenceTransformers downloads the pinned public
`sentence-transformers/all-MiniLM-L6-v2` checkpoint on first use (roughly 90 MB),
then reuses `data/cache/all-MiniLM-L6-v2/`. Copy that cache for offline use and set
`ALLOW_DOWNLOAD=False` in the notebook to require local-only execution. The cache
is ignored by Git. No account, API key, GPU or paid inference service is needed.

The existing `dataset_cache.py` handles the previously documented CSV datasets;
it does not download this model. The model setup is inside SentenceTransformers.
Project outputs live under `outputs/sentence_transformers/` and
`outputs/pydantic/` and are replaced on a normal rerun.

Both notebooks passed fresh-kernel execution on Python 3.9.6: 12 code cells for
SentenceTransformers and 13 for Pydantic. The semantic-search run used the cached
pinned checkpoint; no fallback encoder was substituted. `pip check` reported no
broken requirements. Executed copies are under `outputs/validation/`.
