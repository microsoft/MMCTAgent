#!/bin/bash

set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$(realpath "$script_dir/../infra_config.yaml")"

# Load env vars
source "$script_dir/00-setup-env-vars.sh"

# Function to get value from YAML
get_yaml_value() {
    python -c "import yaml, sys; print(yaml.safe_load(sys.stdin.read())$1)" < "$CONFIG_FILE"
}

# Check if managed identity creation is enabled
if [[ "$(get_yaml_value "['midentityCreation']['enabled']")" != "True" ]]; then
  echo "🔒 Managed identity creation disabled in config. Exiting."
  exit 0
fi

# Fetch subscription ID
subscriptionId=$(az account show --query id -o tsv)
echo "🧾 Subscription ID: $subscriptionId"

# Check/create identity
echo "🔍 Checking if identity '$identityName' exists in resource group '$resourceGroup'..."

if identityExists=$(az identity show \
  --name "$identityName" \
  --resource-group "$resourceGroup" \
  --query "name" \
  --output tsv 2>/dev/null); then
  echo "✅ Identity '$identityName' already exists. Skipping creation."
else
  echo "🆕 Identity does not exist. Creating identity '$identityName'..."
  az identity create \
    --name "$identityName" \
    --resource-group "$resourceGroup"
  echo "✅ Identity '$identityName' created successfully."
fi

clientId=$(az identity show --name "$identityName" --resource-group "$resourceGroup" --query clientId -o tsv)

# Define mapping: yamlKey => (envVar, role, resourceType)
declare -A yaml_to_config_map=(
  ["storageAccount"]="storageAccountName|Storage Blob Data Contributor|Microsoft.Storage/storageAccounts"
  ["azureOpenAIService"]="azureOpenAIName|Cognitive Services OpenAI User|Microsoft.CognitiveServices/accounts"
  ["aiSearchService_data"]="aiSearchServiceName|Search Index Data Contributor|Microsoft.Search/searchServices"
  ["aiSearchService_service"]="aiSearchServiceName|Search Service Contributor|Microsoft.Search/searchServices"
  ["azureSpeechService"]="azureSpeechServiceName|Cognitive Services Speech Contributor|Microsoft.CognitiveServices/accounts"
  ["containerRegistry"]="containerRegistryName|AcrPull|Microsoft.ContainerRegistry/registries"
  ["eventHubService"]="eventhubName|Azure Event Hubs Data Owner|Microsoft.EventHub/namespaces"
)

# Assign roles only if enabled in YAML and variable is non-empty
for yamlKey in "${!yaml_to_config_map[@]}"; do
  # Map AI Search service entries to the correct YAML key
  if [[ "$yamlKey" == "aiSearchService_data" || "$yamlKey" == "aiSearchService_service" ]]; then
    configKey="aiSearchService"
  else
    configKey="$yamlKey"
  fi
  
  isEnabled=$(get_yaml_value "['deployInfra']['$configKey']")
  IFS='|' read -r varName role resourceType <<< "${yaml_to_config_map[$yamlKey]}"
  resourceName="${!varName}"

  if [[ "$isEnabled" == "True" ]]; then
    if [[ -z "$resourceName" ]]; then
      echo "⚠️ Variable for $yamlKey ($varName) is empty. Skipping role assignment."
      continue
    fi

    # Check if the resource exists
    echo "🔍 Checking if resource '$resourceName' of type '$resourceType' exists..."
    if az resource show \
        --name "$resourceName" \
        --resource-group "$resourceGroup" \
        --resource-type "$resourceType" \
        --query "id" \
        --output tsv >/dev/null 2>&1; then

      RESOURCE_SCOPE="/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/$resourceType/$resourceName"
      echo "🔐 Assigning role '$role' on scope '$RESOURCE_SCOPE'"
      
      # Try standard assignment first
      if ! az role assignment create --assignee "$clientId" --role "$role" --scope "$RESOURCE_SCOPE" 2>/dev/null; then
        echo "⚠️ Standard assignment failed. Trying with --assignee-object-id..."
        
        # Get the managed identity object ID
        objectId=$(az identity show --name "$identityName" --resource-group "$resourceGroup" --query principalId -o tsv)
        
        if ! az role assignment create --assignee-object-id "$objectId" --assignee-principal-type "ServicePrincipal" --role "$role" --scope "$RESOURCE_SCOPE" 2>/dev/null; then
          echo "❌ Role assignment failed for $role on $resourceName"
          echo "📝 Manual assignment needed: assign '$role' to managed identity '$identityName' on resource '$resourceName'"
        else
          echo "✅ Role assignment successful using object ID"
        fi
      else
        echo "✅ Role assignment successful"
      fi
    else
      echo "🚫 Resource '$resourceName' does not exist. Skipping role assignment."
    fi
  else
    echo "⏭️ $yamlKey deployment not enabled. Skipping."
  fi
done