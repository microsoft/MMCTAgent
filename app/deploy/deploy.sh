#!/bin/bash
# =============================================================================
# deploy.sh — Deploy MMCT FastAPI Server using Bicep
#
# Usage:
#   ./deploy.sh                          # Build + deploy
#   ./deploy.sh --deploy-only            # Deploy only (image already in ACR)
#   ./deploy.sh --what-if                # Preview changes without deploying
#   ./deploy.sh --params custom.bicepparam  # Use custom parameters file
#
# Prerequisites:
#   - Azure CLI (az) logged in
#   - Docker (unless --deploy-only)
#   - NEO4J_PASSWORD environment variable set
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(realpath "$SCRIPT_DIR/..")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"

BICEP_FILE="$SCRIPT_DIR/main.bicep"
PARAMS_FILE="$SCRIPT_DIR/main.bicepparam"
DEPLOY_ONLY=false
WHAT_IF=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --deploy-only) DEPLOY_ONLY=true ;;
    --what-if) WHAT_IF=true ;;
    --params=*) PARAMS_FILE="${arg#*=}" ;;
    --help|-h)
      head -12 "$0" | tail -10
      exit 0
      ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# Read resource group and ACR from bicepparam (grep for the values)
RESOURCE_GROUP=$(grep -oP "(?<=param location = ').*(?=')" "$PARAMS_FILE" | head -1)
# We need to get RG from the deployment target; parse it from params or use default
RG=$(grep -oP "(?<=DefaultResourceGroup-).*" "$PARAMS_FILE" | head -1 || true)
RESOURCE_GROUP="DefaultResourceGroup-CCAN"
ACR_NAME=$(grep -oP "(?<=param containerRegistryName = ').*(?=')" "$PARAMS_FILE")
IMAGE_TAG=$(grep -oP "(?<=param imageTag = ').*(?=')" "$PARAMS_FILE")
IMAGE_NAME=$(grep -oP "(?<=param imageName = ').*(?=')" "$PARAMS_FILE")

CONTAINER_REGISTRY="${ACR_NAME}.azurecr.io"
BASE_IMAGE="${CONTAINER_REGISTRY}/mmct-base:${IMAGE_TAG}"
MAIN_APP_IMAGE="${CONTAINER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================="
echo "  MMCT FastAPI → Azure Container Apps (Bicep)"
echo "============================================="
echo ""
echo "  Resource Group:  $RESOURCE_GROUP"
echo "  Registry:        $CONTAINER_REGISTRY"
echo "  Image:           $MAIN_APP_IMAGE"
echo "  Bicep:           $BICEP_FILE"
echo "  Params:          $PARAMS_FILE"
echo ""

# Step 1: Build and push Docker images
if [[ "$DEPLOY_ONLY" == "false" && "$WHAT_IF" == "false" ]]; then
  echo "============================================="
  echo "  Step 1: Build & Push Docker Images"
  echo "============================================="
  az acr login --name "$ACR_NAME"

  echo "🔨 Building base image..."
  docker build -f "$PROJECT_ROOT/Dockerfile.base" -t "$BASE_IMAGE" "$PROJECT_ROOT"
  docker push "$BASE_IMAGE"

  echo "🔨 Building app image..."
  docker build -f "$APP_DIR/Dockerfile.main" -t "$MAIN_APP_IMAGE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" "$APP_DIR"
  docker push "$MAIN_APP_IMAGE"
  echo "✅ Images pushed."
else
  echo "⏭️  Skipping build"
fi

# Step 2: Deploy with Bicep
echo ""
echo "============================================="
if [[ "$WHAT_IF" == "true" ]]; then
  echo "  Step 2: What-If Preview"
  echo "============================================="
  az deployment group what-if \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "$BICEP_FILE" \
    --parameters "$PARAMS_FILE"
else
  echo "  Step 2: Deploy Bicep Template"
  echo "============================================="
  az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "$BICEP_FILE" \
    --parameters "$PARAMS_FILE" \
    --query "properties.outputs" \
    --output table

  # Show results
  echo ""
  echo "============================================="
  echo "  ✅ Deployment Complete!"
  echo "============================================="
  FQDN=$(az deployment group show \
    --resource-group "$RESOURCE_GROUP" \
    --name main \
    --query "properties.outputs.fqdn.value" -o tsv 2>/dev/null || echo "<pending>")
  echo ""
  echo "  App URL:    https://$FQDN"
  echo "  Docs:       https://$FQDN/docs"
  echo "  Health:     https://$FQDN/health"
  echo ""
  echo "  Scaling:    3 → 100 replicas (auto, HTTP-based)"
  echo ""
fi
