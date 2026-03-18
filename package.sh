#!/usr/bin/env bash
set -euo pipefail

uv build
uv run --group publish twine upload --repository pypi dist/*
