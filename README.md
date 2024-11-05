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


# Contributing 

Contributions to supercontrast are welcome. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines.
