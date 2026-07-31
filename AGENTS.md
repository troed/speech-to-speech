# Repository Instructions

- Never include `codex` in branch names or pull request titles.
- Keep release pull requests focused on version metadata and release documentation.
- Do not commit local build artifacts such as `dist/`, `build/`, or generated wheel/sdist files.

## Code Exploration Policy

Always use jCodemunch-MCP tools — never fall back to Read, Grep, Glob, or Bash for code exploration.

- Before reading a file: use get_file_outline or get_file_content
- Before searching: use search_symbols or search_text
- Before exploring structure: use get_file_tree or get_repo_outline
- Call resolve_repo with the current directory first; if not indexed, call index_folder.

## Branch Workflow

Development happens on the `devel` branch. `main` is reserved for releases.

- All day-to-day work, commits, and experiments go on `devel`.
- When a set of changes represents a cohesive new feature or major improvement, open a pull request from `devel` to `main`.
- Merge to `main` only when preparing a release — then follow the PyPI publishing workflow below.
- `main` should always be in a releasable state.

## Publishing to PyPI

PyPI publishing is handled by GitHub Actions in `.github/workflows/publish.yml`. The workflow runs on pushed tags that match `v*`, builds the package with `uv build`, checks the artifacts with `twine check --strict`, and publishes through the configured `pypi` environment.

To prepare a release:

1. Confirm the intended version is not already published on PyPI.
2. Bump `version` in `pyproject.toml`.
3. Bump `__version__` in `src/speech_to_speech/__init__.py`.
4. Open and merge a pull request with only the release preparation changes.

To publish after the release PR is merged:

1. Update `main` locally: `git checkout main && git pull origin main`.
2. Create an annotated tag for the version: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
3. Push the tag: `git push origin vX.Y.Z`.
4. Watch the `Publish` GitHub Actions workflow complete successfully.
5. Verify the new version appears at `https://pypi.org/project/speech-to-speech/`.

Only upload manually if the GitHub Actions workflow is unavailable and the maintainers have explicitly chosen that fallback.
