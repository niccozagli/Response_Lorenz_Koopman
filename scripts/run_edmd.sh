#!/bin/bash

set -e
set -u

echo "Running EDMD analysis..."
poetry run python scripts/perform_edmd.py
echo "Done."

