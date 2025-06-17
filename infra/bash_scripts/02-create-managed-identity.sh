#!/bin/bash

# Variables
source 00-setup-env-vars.sh

subscriptionId=$(az account show --query id -o tsv)

echo "Subscription id: $subscriptionId"

# set the subscription name if subscription not set
# az account set --subscription ""

# Check if the identity exists
echo "Checking if the identity '$identityName' exists in resource group '$resourceGroup'..."
identityExists=$(az identity show \
  --name "$identityName" \
  --resource-group "$resourceGroup" \
  --query "name" \
  --output tsv 2>/dev/null)

# Create the identity if it does not exist
if [ -z "$identityExists" ]; then
  echo "Identity does not exist. Creating identity '$identityName'..."
  az identity create \
    --name "$identityName" \
    --resource-group "$resourceGroup"
  echo "Identity '$identityName' created successfully."
else
  echo "Identity '$identityName' already exists. Skipping creation."
fi

clientId=$(az identity show --name $identityName --resource-group $resourceGroup --query clientId -o tsv)

# Declaring the roles and assigning the them to the managed identity resource
declare -A resources=(
  ["Storage Blob Data Contributor"]="Microsoft.Storage/storageAccounts/$storageAccountName"
  ["Cognitive Services OpenAI User"]="Microsoft.CognitiveServices/accounts/$azureOpenAIName"
  ["Search Index Data Contributor"]="Microsoft.Search/searchServices/$aiSearchServiceName"
  ["Search Service Contributor"]="Microsoft.Search/searchServices/$aiSearchServiceName"
  ["Azure Event Hubs Data Owner"]="Microsoft.EventHub/namespaces/$eventhubName"
  ["Cognitive Services Speech Contributor"]="Microsoft.CognitiveServices/accounts/$azureSpeechServiceName"
  ["AcrPull"]="Microsoft.ContainerRegistry/registries/$containerRegistryName"
)

for role in "${!resources[@]}"; do
  RESOURCE_SCOPE="subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/${resources[$role]}"
  echo "Assigning role '$role' to identity '$identityName' on scope '$RESOURCE_SCOPE'"
  az role assignment create --assignee $clientId --role "$role" --scope $RESOURCE_SCOPE
done