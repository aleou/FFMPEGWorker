#!/bin/bash
# Script to fully rebuild Docker image with cache clearing
# Usage: Run this on your Ubuntu VM after pulling latest code

set -e  # Exit on any error

echo "================================================"
echo "🚀 FFMPEGWorker - Full Rebuild & Push Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get version from docker-bake.hcl
VERSION=$(grep 'default = "' docker-bake.hcl | grep RELEASE_VERSION -A 1 | tail -1 | sed 's/.*"\(.*\)".*/\1/')
echo -e "${GREEN}📦 Version: ${VERSION}${NC}"
echo ""

# Step 1: Stop all running containers
echo -e "${YELLOW}⏹️  Stopping all running containers...${NC}"
docker stop $(docker ps -aq) 2>/dev/null || echo "No containers to stop"
echo ""

# Step 2: Remove old images
echo -e "${YELLOW}🗑️  Removing old aleou/ffmpeg-worker images...${NC}"
docker rmi -f $(docker images 'aleou/ffmpeg-worker' -q) 2>/dev/null || echo "No images to remove"
echo ""

# Step 3: Clear buildx cache
echo -e "${YELLOW}🧹 Clearing Docker buildx cache...${NC}"
docker buildx prune -a -f
echo ""

# Step 4: Clear system cache
echo -e "${YELLOW}🧹 Clearing Docker system cache...${NC}"
docker system prune -a -f
echo ""

# Step 5: Show disk usage
echo -e "${YELLOW}💾 Current Docker disk usage:${NC}"
docker system df
echo ""

# Step 6: Rebuild with no cache
echo -e "${GREEN}🔨 Building fresh images with --no-cache...${NC}"
echo "This will take 10-15 minutes (downloading models + compiling)..."
echo ""
docker buildx bake --no-cache --push

# Step 7: Verify push
echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo -e "${YELLOW}📋 Verify on Docker Hub:${NC}"
echo "   https://hub.docker.com/r/aleou/ffmpeg-worker/tags"
echo ""
echo -e "${YELLOW}🔍 Check image digest:${NC}"
docker images aleou/ffmpeg-worker --digests
echo ""
echo -e "${GREEN}🎉 Done! New image pushed with version ${VERSION}${NC}"
echo ""
echo -e "${YELLOW}⚠️  Next steps:${NC}"
echo "1. Update RunPod template to use: aleou/ffmpeg-worker:${VERSION}-serverless"
echo "2. Or use digest for guaranteed exact image: aleou/ffmpeg-worker@sha256:xxxxx"
echo "3. Run test job and check logs for real error messages (no more '%s')"
echo ""
