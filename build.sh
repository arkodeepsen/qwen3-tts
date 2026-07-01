#!/bin/bash
set -e
GITHUB_USERNAME="${GITHUB_USERNAME:-arkodeepsen}"
IMAGE_NAME="qwen3-tts"
REGISTRY="ghcr.io/${GITHUB_USERNAME}"
TAG="${1:-latest}"

echo "Building ${REGISTRY}/${IMAGE_NAME}:${TAG}..."
docker build -f Dockerfile -t "${IMAGE_NAME}:${TAG}" -t "${REGISTRY}/${IMAGE_NAME}:${TAG}" .

echo "✅ Built. To push:"
echo "  docker login ghcr.io -u ${GITHUB_USERNAME}"
echo "  docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}"
