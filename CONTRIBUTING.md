# Contributing

We welcome contributions to the supercontrast! To get started, please follow these steps:

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

### 3. Make Your Changes
Create a new branch to make your changes. Use a descriptive name for your branch, such as `feature/add-new-feature` or `bugfix/fix-issue-123`

```
git checkout -b feature/add-new-feature

```

### 4. Run Linting
To maintain code quality and consistency, run linting tools:

```
black .
isort .
```

### 5. Commit Your Changes
After making and verifying changes, commit them with a clear and concise message:

```
git add .
git commit -m "Add new feature: description of feature"
```

### 6. Run Tests
Run the test suite to ensure everything is working correctly:
```
pytest
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
### Issue Reporting

If you find a bug or would like to request a feature, please open an issue on GitHub. Provide as much detail as possible, including steps to reproduce the problem if applicable.

### Documentation Improvements

We welcome contributions to our documentation. If you find outdated information, unclear explanations, or think a section could use more examples, feel free to submit a pull request.

### Community Guidelines

When contributing to this project, please adhere to our code of conduct. Be respectful of others' opinions and contributions, and keep the discussions focused on making the project better.
