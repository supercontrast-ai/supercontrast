# Supercontrast
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


# Contributing to Supercontrast

We welcome contributions to the Supercontrast project! To get started, please follow these steps:

### 1. Fork the Repository

First, fork the repository on GitHub to create your own copy. Then, clone your fork to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/supercontrast.git
```

### 2. Install the Package
Navigate to the project directory and install the package in development mode:

``` 
cd supercontrast
pip install -e .[dev]

```

### 3. Run Linting
To maintain code quality and consistency, run linting tools:

```
black .
isort .
```
### 4. Run Tests
Run the test suite to ensure everything is working correctly:
```
pytest

```
### 5. Make Your Changes
Make your changes in a new branch. Use a descriptive name for your branch, such as feature/add-new-feature or bugfix/fix-issue-123

```
git checkout -b feature/add-new-feature

```


### 6. Commit Your Changes
After making changes, commit them with a clear and concise message:

```
git add .
git commit -m "Add new feature: description of feature"
```

### 7. Push to Your Fork
Push your changes to your fork on GitHub:
```
git push origin feature/add-new-feature
```

### 8. Submit a Pull Request (PR)
Go to the original repository on GitHub and submit a pull request from your branch. Provide a detailed description of your changes, including the purpose and any relevant details. You can use the following command to create the PR via GitHub CLI:

```
gh pr create --title "Add new feature: description" --body "Detailed description of the changes."

```


### For more detailed contribution guidelines, please see the [CONTRIBUTING.md](CONTRIBUTING.md) file.



