#!/bin/bash

# Manual role assignment helper script
# Run this from Azure Cloud Shell or a compliant device

set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load env vars
source "$script_dir/00-setup-env-vars.sh"

# Get managed identity details
subscriptionId=$(az account show --query id -o tsv)
objectId=$(az identity show --name "$identityName" --resource-group "$resourceGroup" --query principalId -o tsv)

echo "🔧 Manual Role Assignment Commands"
echo "=================================="
echo "Managed Identity: $identityName"
echo "Object ID: $objectId"
echo "Resource Group: $resourceGroup"
echo ""

# Generate role assignment commands
echo "# Copy and run these commands from a compliant device:"
echo ""

# Azure OpenAI
if [[ -n "$azureOpenAIName" ]]; then
    echo "# Azure OpenAI Role Assignment"
    echo "az role assignment create \\"
    echo "  --assignee-object-id '$objectId' \\"
    echo "  --assignee-principal-type 'ServicePrincipal' \\"
    echo "  --role 'Cognitive Services OpenAI User' \\"
    echo "  --scope '/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.CognitiveServices/accounts/$azureOpenAIName'"
    echo ""
fi

# Storage Account
if [[ -n "$storageAccountName" ]]; then
    echo "# Storage Account Role Assignment"
    echo "az role assignment create \\"
    echo "  --assignee-object-id '$objectId' \\"
    echo "  --assignee-principal-type 'ServicePrincipal' \\"
    echo "  --role 'Storage Blob Data Contributor' \\"
    echo "  --scope '/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.Storage/storageAccounts/$storageAccountName'"
    echo ""
fi

# AI Search
if [[ -n "$aiSearchServiceName" ]]; then
    echo "# AI Search Role Assignments (Both roles needed)"
    echo "az role assignment create \\"
    echo "  --assignee-object-id '$objectId' \\"
    echo "  --assignee-principal-type 'ServicePrincipal' \\"
    echo "  --role 'Search Index Data Contributor' \\"
    echo "  --scope '/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.Search/searchServices/$aiSearchServiceName'"
    echo ""
    echo "az role assignment create \\"
    echo "  --assignee-object-id '$objectId' \\"
    echo "  --assignee-principal-type 'ServicePrincipal' \\"
    echo "  --role 'Search Service Contributor' \\"
    echo "  --scope '/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.Search/searchServices/$aiSearchServiceName'"
    echo ""
fi

# Speech Service
if [[ -n "$azureSpeechServiceName" ]]; then
    echo "# Speech Service Role Assignment"
    echo "az role assignment create \\"
    echo "  --assignee-object-id '$objectId' \\"
    echo "  --assignee-principal-type 'ServicePrincipal' \\"
    echo "  --role 'Cognitive Services Speech Contributor' \\"
    echo "  --scope '/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.CognitiveServices/accounts/$azureSpeechServiceName'"
    echo ""
fi

# Event Hub
if [[ -n "$eventhubName" ]]; then
    echo "# Event Hub Role Assignment"
    echo "az role assignment create \\"
    echo "  --assignee-object-id '$objectId' \\"
    echo "  --assignee-principal-type 'ServicePrincipal' \\"
    echo "  --role 'Azure Event Hubs Data Owner' \\"
    echo "  --scope '/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.EventHub/namespaces/$eventhubName'"
    echo ""
fi

# Container Registry
if [[ -n "$containerRegistryName" ]]; then
    echo "# Container Registry Role Assignment"
    echo "az role assignment create \\"
    echo "  --assignee-object-id '$objectId' \\"
    echo "  --assignee-principal-type 'ServicePrincipal' \\"
    echo "  --role 'AcrPull' \\"
    echo "  --scope '/subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.ContainerRegistry/registries/$containerRegistryName'"
    echo ""
fi

echo "📋 Save these commands and run them from Azure Cloud Shell or a compliant device."