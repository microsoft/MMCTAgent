#!/bin/bash

set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$(realpath "$script_dir/../infra_config.yaml")"
project_root="$(realpath "$script_dir/../..")"
# Load env vars
source "$script_dir/00-setup-env-vars.sh"

# Function to get value from YAML
get_yaml_value() {
    python -c "import yaml, sys; print(yaml.safe_load(sys.stdin.read())$1)" < "$CONFIG_FILE"
}

# Check if building and pushing of docker images is enabled
if [[ "$(get_yaml_value "['buildAndPushImagesToACR']['enabled']")" != "True" ]]; then
  echo "Building of docker images is disabled in config. Exiting."
  exit 0
fi

# Function to check if the Azure Container Registry exists
check_acr_exists() {
  echo "🔍 Checking if Azure Container Registry '$containerRegistryName' exists in resource group '$resourceGroup'..."
  if az resource show \
    --name "$containerRegistryName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.ContainerRegistry/registries" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Azure Container Registry '$containerRegistryName' exists."
  else
    echo "❌ Azure Container Registry '$containerRegistryName' does not exist in resource group '$resourceGroup'. Exiting."
    exit 1
  fi
}

# Function to log in to the Azure Container Registry
login_to_acr() {
  echo "🔐 Logging into Azure Container Registry '$containerRegistryName'..."
  az acr login --name "$containerRegistryName"
  echo "✅ Logged in to Azure Container Registry '$containerRegistryName'."
}


check_acr_exists
login_to_acr


build_main_app="$(get_yaml_value "['buildAndPushImagesToACR']['buildMainApp']")"
build_ingest="$(get_yaml_value "['buildAndPushImagesToACR']['buildIngestionConsumer']")"

# Build base image only if needed
if [[ "$build_main_app" == "True" || "$build_ingest" == "True" ]]; then
  echo "🚧 Building base image..."
  docker build -f "$project_root/Dockerfile.base" \
    -t "$baseImage" \
     "$project_root"
  echo "✅ Built base image."
  echo "pushing the docker base image $baseImage"
  docker push "$baseImage"
  echo "✅ Successfully built and pushed $baseImage'."
fi

# Build and push main app
if [[ "$build_main_app" == "True" ]]; then
  echo "🚀 Building Main App image..."
  docker build \
    -f "$project_root/app/Dockerfile.main" \
    -t "${containerRegistryName}.azurecr.io/main-app:${imageTag}" \
    --build-arg BASE_IMAGE="$baseImage"  \
    "$project_root/app"

  echo "📤 Pushing Main App image..."
  docker push "${containerRegistryName}.azurecr.io/main-app:${imageTag}"
fi

# Build and push ingestion consumer
if [[ "$build_ingest" == "True" ]]; then
  echo "🚀 Building Ingestion Consumer image..."
  docker build \
    -f "$project_root/app/Dockerfile.ingestion_consumer" \
    -t "${containerRegistryName}.azurecr.io/ingestion-consumer:${imageTag}" \
    --build-arg BASE_IMAGE="$baseImage"  \
    "$project_root/app"

  echo "📤 Pushing Ingestion Consumer image..."
  docker push "${containerRegistryName}.azurecr.io/ingestion-consumer:${imageTag}"
fi