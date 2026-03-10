#!/usr/bin/env bash
# Build the MMCT app Docker image with automatic patch version bump.
# Run from the repository root: ./app/build.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION_FILE="$REPO_ROOT/app/version.py"
BASE_IMAGE="${BASE_IMAGE:-mmct-base:latest}"

# --- Bump patch version in version.py on disk ---
CURRENT=$(grep '_DEFAULT_VERSION' "$VERSION_FILE" | head -1 | grep -oP '\d+\.\d+\.\d+')
MAJOR=$(echo "$CURRENT" | cut -d. -f1)
MINOR=$(echo "$CURRENT" | cut -d. -f2)
PATCH=$(echo "$CURRENT" | cut -d. -f3)
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="${MAJOR}.${MINOR}.${NEW_PATCH}"

sed -i "s/_DEFAULT_VERSION = \"${CURRENT}\"/_DEFAULT_VERSION = \"${NEW_VERSION}\"/" "$VERSION_FILE"
echo "Version bumped: ${CURRENT} → ${NEW_VERSION}"

# --- Build the Docker image ---
DOCKER_BUILDKIT=0 docker build \
  -f "$REPO_ROOT/app/Dockerfile.main" \
  -t mmct-lively-fastapi:latest \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  "$REPO_ROOT"
