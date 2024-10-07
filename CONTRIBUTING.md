# Contributing to Supercontrast

We welcome contributions to the Supercontrast project! To get started, please follow these steps:

## Getting Started

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
















