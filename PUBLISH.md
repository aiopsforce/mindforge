# Publishing MindForge to PyPI

This document outlines the steps to publish the `MindForge` package to PyPI.

## Prerequisites

Ensure you have the following installed:
- `build`
- `twine`

You can install them via pip:
```bash
pip install build twine
```

## Steps

1.  **Update Version**: Ensure the version number in `pyproject.toml` is correct and updated for the new release.

2.  **Build the Package**: Run the following command from the root of the repository to build the source distribution and wheel:
    ```bash
    python -m build
    ```
    This will create a `dist/` directory containing the `.tar.gz` and `.whl` files.

3.  **Check the Package**: Use `twine` to check the build artifacts for any issues:
    ```bash
    twine check dist/*
    ```

4.  **Upload to TestPyPI (Optional but Recommended)**: uploading to TestPyPI first allows you to verify that everything looks correct without affecting the production index.
    ```bash
    twine upload --repository testpypi dist/*
    ```
    You will need an account on [TestPyPI](https://test.pypi.org/).

5.  **Upload to PyPI**: Once verified, upload the package to the real PyPI:
    ```bash
    twine upload dist/*
    ```
    You will need an account on [PyPI](https://pypi.org/).

## Automation (CI/CD)

It is recommended to automate this process using GitHub Actions or a similar CI/CD tool. A typical workflow would:
1.  Trigger on a new tag (e.g., `v1.0.0`).
2.  Run tests.
3.  Build the package.
4.  Publish to PyPI using a repository secret (API token).
