# supercontrast
<h4 align="center">
    <a href="https://pypi.org/project/supercontrast/" target="_blank">
        <img src="https://img.shields.io/pypi/v/supercontrast.svg" alt="PyPI Version">
    </a>
    <a href="https://www.ycombinator.com/companies/supercontrast">
        <img src="https://img.shields.io/badge/Y%20Combinator-F24-orange?style=flat-square" alt="Y Combinator F24">
    </a>
    <a href="https://discord.gg/R9TSAc23">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
    <a href="https://docs.supercontrast.com/" target="_blank">
        <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation Status">
    </a>
</h4>

`supercontrast` is a package for easily running machine learning models from a variety of providers in a unified interface. We're adding more tasks and providers all the time, and would love help from the community to add more!


We currently support the following tasks:

- **OCR**
- **Sentiment Analysis**
- **Transcription**
- **Translation**

From some of the most popular providers:

- **AWS**
- **Azure**
- **GCP**
- **OpenAI**
- **Anthropic**
- **...and more!**


## Installation

```bash
pip install supercontrast
```

**NOTE:** `supercontrast` is supported on Python 3.12, it may be unstable on other versions. If you have conda installed, you can create an environment with the required packages using the `environment.yml` file:

```bash
conda env create -f environment.yml
```

## Usage

```python
from supercontrast import (
    Provider,
    SentimentAnalysisRequest,
    SuperContrastClient,
    Task,
)

# Sending a Sentiment Analysis Request to AWS
client = SuperContrastClient(task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS])
input_text = "I love programming in Python!"
response, metadata = client.request(SentimentAnalysisRequest(text=input_text))
```

For more examples of how to use `supercontrast`, refer to [examples.py](examples/examples.py) in the [examples](examples/) folder.

## Contributing

We welcome contributions to the project! To contribute, please follow these steps:

### 1. Clone the repo

```bash
git clone https://github.com/supercontrast/supercontrast.git
```

### 2. Install package

```bash
pip install -e .[dev]
```

### 3. Run linting

```bash
black .
isort .
```

### 4. Run tests

```bash
pytest -k <test_name>
```

### 5. Submit a PR

Submit a PR to the main branch! We will review and merge your PR.









