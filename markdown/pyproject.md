[tool.poetry]
name = "centralized-nlp-package"
version = "0.1.0"
description = "A centralized, modular Python package for NLP pipelines on Databricks."
authors = ["santhosh <skumarr53@gmail.com>"]
readme = "README.md"
homepage = "https://github.com/Skumarr53/cnp"
repository = "https://github.com/Skumarr53/cnp"
license = "MIT"

[tool.poetry.dependencies]
python = ">=3.9, <3.11"
# Runtime dependencies
loguru = "^0.7.2"
hydra-core = "^1.3"
python-dotenv = "^1.0.1"
numpy = "^1.24.4"
cryptography = "^43.0.1"
gensim = "^4.3.3"
cython = "^3.0.11"
spacy = "^3.0.4"
thinc = "^8.1.7"
pandas = "^2.0.0"
snowflake-connector-python = "^3.12.2"
transformers = "4.46.1"
pyarrow = "^16.0.0"
datasets = "^3.1.0"
evaluate = "^0.4.3"
pyspark = "^3.5.3"
dask = {version = "^2023", extras = ["dataframe", "distributed"]}
torch = "^2.0.0"
cymem = "2.0.8"
scikit-learn = "^1.1.0"
databricks = "^0.2"
databricks-sdk = "^0.38.0"
mlflow-skinny = "2.18.0"
accelerate = "0.26.0"
torchvision = "^0.20.1"
mlflow = "2.18.0"

[tool.poetry.scripts]
centralized_nlp = "centralized_nlp_package.__main__:main"

[tool.poetry.group.dev.dependencies]
black = "^20.0.0"
pytest = "^6.0.0"
sphinx = "^7.1.1"
sphinx-autodoc-typehints = "^2.0"
furo = "^2024.8.6"
flake8 = "^6.0.0"
mypy = "^1.5.0"
tox = "^4.23.2"


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"