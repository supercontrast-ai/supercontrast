# supercontrast
<h4 align="center">
    <a href="https://pypi.org/project/supercontrast/" target="_blank">
        <img src="https://img.shields.io/pypi/v/supercontrast.svg" alt="PyPI Version">
    </a>
    <a href="https://www.ycombinator.com/companies/supercontrast">
        <img src="https://img.shields.io/badge/Y%20Combinator-F24-orange?style=flat-square" alt="Y Combinator F24">
    </a>
    <a href="https://discord.com/invite/G6emkXjAm2">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
    <a href="https://docs.supercontrast.com/" target="_blank">
        <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation Status">
    </a>
</h4>

`supercontrast` is a package for easily running machine learning models from a variety of providers in a unified interface. We're adding more tasks and providers all the time, and would love help from the community to add more!


We currently support the following **Tasks**:

- **OCR**
- **Sentiment Analysis**
- **Transcription**
- **Translation**
- **...and more!**

From some of the most popular **Providers**:

- **AWS**
- **Azure**
- **GCP**
- **OpenAI**
- **Anthropic**
- **...and more!**

If you want the full list of supported tasks and providers, please reference our [docs](https://docs.supercontrast.com/introduction).

## Installation

### pip

```bash
pip install supercontrast
```

### conda

```bash
conda env create -f environment.yml
```

## Additional Requirements

### Python Version
`supercontrast` is supported on **Python 3.12** (other versions may be unstable)

### PDF Processing
If you are processing pdfs, you will need to install `poppler`. We recommend using conda to install it:

```bash
conda install -c conda-forge poppler
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

Contributions to supercontrast are welcome. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines.