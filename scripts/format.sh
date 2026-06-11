#!/usr/bin/env bash
#
# Formats all C++ source files in the project in-place using clang-format,
# according to the rules in the project's .clang-format file.
#
# Usage:
#   scripts/format.sh           Format all files in-place.
#   scripts/format.sh --check   Report unformatted files without changing them
#                               (exits non-zero if any need formatting).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Directories that contain source code
SOURCE_DIRS=(src include tests)

# File extensions to format
EXTENSIONS=(cpp hpp cc cxx h hxx)

if ! command -v clang-format >/dev/null 2>&1; then
  echo "error: clang-format not found in PATH" >&2
  exit 1
fi

CHECK=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK=1
fi

# Build the find expression for the configured extensions
find_args=()
for ext in "${EXTENSIONS[@]}"; do
  find_args+=(-o -name "*.${ext}")
done
# Drop the leading "-o".
find_args=("${find_args[@]:1}")

# Collect all matching files into an array
files=()
while IFS= read -r -d '' file; do
  files+=("$file")
done < <(
  find "${SOURCE_DIRS[@]/#/${PROJECT_ROOT}/}" \
    -type f \( "${find_args[@]}" \) -print0 2>/dev/null
)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No source files found."
  exit 0
fi

if [[ ${CHECK} -eq 1 ]]; then
  status=0
  prefix="${PROJECT_ROOT}/"
  for file in "${files[@]}"; do
    if ! diff -q <(clang-format "$file") "$file" >/dev/null; then
      echo "needs formatting: ${file#"$prefix"}"
      status=1
    fi
  done
  [[ ${status} -eq 0 ]] && echo "All ${#files[@]} files are correctly formatted."
  exit ${status}
fi

clang-format -i "${files[@]}"
echo "Formatted ${#files[@]} files."