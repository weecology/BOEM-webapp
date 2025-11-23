#!/usr/bin/env bash
# Helper script to check that no annotations.csv files are modified in a PR or commit.
# Usage:
#  - From CI: check against a base ref (e.g. git fetch origin main; check_annotations.sh origin/main HEAD)
#  - From pre-commit / local: uses git diff --cached to inspect staged files.
set -euo pipefail

# If two args provided, treat them as base and head for comparison
if [ "$#" -eq 2 ]; then
  base_ref="$1"
  head_ref="$2"
  changed_files=$(git diff --name-only "$base_ref" "$head_ref" || true)
else
  # Default: check staged files (for pre-commit)
  changed_files=$(git diff --cached --name-only || true)
fi

if printf '%s\n' "$changed_files" | grep -qE '(^|/)?annotations\.csv$'; then
  echo "ERROR: You are modifying one or more annotations.csv files:"
  printf '%s\n' "$changed_files" | grep -E '(^|/)?annotations\.csv$' || true
  echo "These files are centrally maintained and should not be changed in PRs."
  echo "If you need to update annotation data, please follow the project's data update process (open an issue or contact the maintainers)."
  exit 1
fi

echo "OK: No annotations.csv files modified."
exit 0
