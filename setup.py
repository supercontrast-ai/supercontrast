from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="supercontrast",
    version="0.0.2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=required,
    author="supercontrast",
    author_email="shravan@supercontrast.com",
    description="supercontrast is a package for unifying machine learning models across providers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/supercontrast-ai/supercontrast",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
