#!/usr/bin/env bash
set -euo pipefail

# Usage: ./push_to_github.sh https://github.com/<user>/<repo>.git
if [ $# -ne 1 ]; then
  echo "Usage: $0 <repo-url>"
  exit 1
fi
REPO_URL="$1"

git init
git add .
git commit -m "Initial commit: review package (scripts+data)"
git branch -M main
git remote add origin "$REPO_URL"
git push -u origin main
