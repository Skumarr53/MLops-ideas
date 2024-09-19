# setup.py

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="centralized_nlp_package",
    version="0.1.0",
    author="Santhosh Kumar",
    author_email="santhosh.kumar3@voya.com",
    description="A centralized, modular Python package for NLP pipelines on Databricks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/centralized_nlp_package",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "dask",
        "pyarrow",
        "gensim",
        "spacy",
        "plotly",
        "umap-learn",
        "loguru",
        "hydra-core",
        # Add other dependencies as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
