#!/bin/bash
# Rebuild and push Docker images with selectable cache strategy.
# Usage:
#   ./rebuild_and_push.sh            # soft rebuild (default)
#   ./rebuild_and_push.sh soft       # same as default
#   ./rebuild_and_push.sh full       # clear caches and rebuild from scratch
#   ./rebuild_and_push.sh --help     # show help

set -euo pipefail

SOFT_ICON="[soft]"
FULL_ICON="[full]"

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================"
echo "FFMPEGWorker - Rebuild & Push Script"
echo "================================================"
echo ""

show_help() {
  cat <<'EOF'
Usage: ./rebuild_and_push.sh [soft|full]

  soft (default)  Rebuild using existing Docker cache (faster)
  full            Stop containers, purge cache, rebuild from scratch
  --help          Display this help message
EOF
}

MODE="${1:-soft}"
case "$MODE" in
  soft|--soft)
    echo -e "${YELLOW}${SOFT_ICON} Soft rebuild selected (cache preserved).${NC}"
    CLEAN_CACHE=false
    NO_CACHE_FLAG=""
    ;;
  full|--full)
    echo -e "${YELLOW}${FULL_ICON} Full rebuild selected (cache will be cleared).${NC}"
    CLEAN_CACHE=true
    NO_CACHE_FLAG="--no-cache"
    ;;
  -h|--help|help)
    show_help
    exit 0
    ;;
  *)
    echo -e "${RED}Unknown option '$MODE'. Use soft, full or --help.${NC}"
    exit 1
    ;;
esac
echo ""

# Extract release version from docker-bake.hcl
VERSION=$(grep -A1 'variable "RELEASE_VERSION"' docker-bake.hcl | grep 'default' | sed 's/.*"\(.*\)".*/\1/' || true)
if [ -z "${VERSION}" ]; then
  echo -e "${YELLOW}Warning: unable to determine RELEASE_VERSION from docker-bake.hcl. Using 'unknown'.${NC}"
  VERSION="unknown"
fi
echo -e "${GREEN}Version detected: ${VERSION}${NC}"
echo ""

if [ "$CLEAN_CACHE" = true ]; then
  echo -e "${YELLOW}${FULL_ICON} Stopping running containers...${NC}"
  docker stop $(docker ps -aq) 2>/dev/null || echo "No containers to stop."
  echo ""

  echo -e "${YELLOW}${FULL_ICON} Removing existing aleou/ffmpeg-worker images...${NC}"
  docker rmi -f $(docker images 'aleou/ffmpeg-worker' -q) 2>/dev/null || echo "No images to remove."
  echo ""

  echo -e "${YELLOW}${FULL_ICON} Pruning Docker buildx cache...${NC}"
  docker buildx prune -a -f
  echo ""

  echo -e "${YELLOW}${FULL_ICON} Pruning Docker system cache...${NC}"
  docker system prune -a -f
  echo ""
else
  echo -e "${YELLOW}${SOFT_ICON} Skipping cache and container cleanup. Use 'full' to force a clean rebuild.${NC}"
  echo ""
fi

echo -e "${YELLOW}Current Docker disk usage:${NC}"
docker system df
echo ""

if [ "$CLEAN_CACHE" = true ]; then
  echo -e "${GREEN}${FULL_ICON} Building images with --no-cache (this may take a while).${NC}"
else
  echo -e "${GREEN}${SOFT_ICON} Building images using existing cache.${NC}"
fi
echo ""

docker buildx bake $NO_CACHE_FLAG --push

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo -e "${YELLOW}Verify on Docker Hub:${NC}"
echo "  https://hub.docker.com/r/aleou/ffmpeg-worker/tags"
echo ""
echo -e "${YELLOW}Check image digests:${NC}"
docker images aleou/ffmpeg-worker --digests
echo ""
echo -e "${GREEN}Done! New image pushed with version ${VERSION}.${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update RunPod template to use: aleou/ffmpeg-worker:${VERSION}-serverless"
echo "2. Or reference the image digest for a fixed deployment"
echo "3. Run a test job and confirm logs look healthy"
echo ""
