# Windows pip recovery hotfix

The Windows launchers now validate `pip` inside an existing `.venv` before upgrading dependencies.

If the virtual environment contains a partial/corrupted pip installation (for example `ModuleNotFoundError: No module named 'pip._internal.utils'`), the launcher:

1. tries `python -m ensurepip --upgrade`;
2. validates pip again;
3. if pip is still broken, deletes and recreates `.venv` automatically;
4. resumes the normal dependency installation.

This does not change solver, scoring, vision, strategy, persistence, or research logic.
