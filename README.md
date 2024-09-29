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
</h4>

`supercontrast` is a package for easily running models from a variety of providers.

## Installation

```bash
pip install supercontrast
```

## Usage

```python
from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task.task_enum import Task
from supercontrast.task.types.sentiment_analysis_types import SentimentAnalysisRequest

# Sending a Sentiment Analysis Request to AWS
client = supercontrast_client(task=Task.SENTIMENT_ANALYSIS, providers=[Provider.AWS], optimizer=None)
input_text = "I love this product!"
response = client.request(SentimentAnalysisRequest(text=input_text))
```

For more examples, see the [examples](examples/examples.py) folder.

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
pytest
```

### 5. Submit a PR

Submit a PR to the main branch! We will review and merge your PR.









