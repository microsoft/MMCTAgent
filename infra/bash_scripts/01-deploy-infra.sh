#!/bin/bash
set -e
# Get directory of the current script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the env vars using absolute path
source "$script_dir/00-setup-env-vars.sh"

# ------------------ DEPLOY RESOURCES -------------

# ----- 1. STORAGE ACCOUNT ------- 
# Check if storage account exists
echo "Checking if storage account '$storageAccountName' exists..."

if az resource show \
    --name "$storageAccountName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.Storage/storageAccounts" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $storageAccountName"
else
    echo "⚙️ Resource does not exist, creating it..."
    # Deploy storage accounts
    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$storageAccountTemplateFile" \
      --parameters storageAccountName=$storageAccountName \
      --debug
fi

# ------ 2. AI Search Serice --------
# check if ai search service exists

echo "Check if AI search service $aiSearchServiceName exists...."

if az resource show \
    --name "$aiSearchServiceName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.Search/searchServices" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $aiSearchServiceName"
else
    echo "⚙️ Resource does not exist, creating it..."

    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$aiSearchServiceTemplateFile" \
      --parameters aiSearchServiceName=$aiSearchServiceName \
      --debug
fi

# ------ 3. Azure Speech Service --------
# check if azure speech service exists

echo "Check if azure speech service $azureSpeechServiceName exists...."

if az resource show \
    --name "$azureSpeechServiceName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.CognitiveServices/accounts" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $azureSpeechServiceName"
else
    echo "⚙️ Resource does not exist, creating it..."

    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$azureSpeechServiceTemplateFile" \
      --parameters azureSpeechServiceName=$azureSpeechServiceName \
            azureSpeechServiceRegion=$azureSpeechServiceRegion \
      --debug
fi

# ------ 4. Azure Container Registry --------
# check if azure container registrye exists

echo "Check if azure container registry $containerRegistryName exists...."

if az resource show \
    --name "$containerRegistryName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.ContainerRegistry/registries" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $containerRegistryName"
else
    echo "⚙️ Resource does not exist, creating it..."

    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$containerRegistryTemplateFile" \
      --parameters containerRegistryName=$containerRegistryName \
      --debug
fi

# ------ 5. Azure App Service Plan --------
# check if azure app service plan exists

echo "Check if azure premium app service plan $aspPremiumName exists...."

if az resource show \
    --name "$aspPremiumName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.Web/serverfarms" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $aspPremiumName"
else
    echo "⚙️ Resource does not exist, creating it..."
 
    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$aspPremiumTemplateFile" \
      --parameters aspPremiumName=$aspPremiumName \
      --debug
fi


# check if azure app service plan exists

echo "Check if azure basic app service plan $aspBasicName exists...."

if az resource show \
    --name "$aspBasicName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.Web/serverfarms" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $aspBasicName"
else
    echo "⚙️ Resource does not exist, creating it..."

    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$aspBasicTemplateFile" \
      --parameters aspBasicName=$aspBasicName \
      --debug
fi

#------6. Azure Event Hub  --------
# check if azure event hub  exists

echo "Check if azure evnet hub  $eventhubName exists...."

if az resource show \
    --name "$eventhubName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.EventHub/namespaces" \
    --api-version "2024-05-01-preview"\
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $eventhubName"
else
    echo "⚙️ Resource does not exist, creating it..."
    
    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$eventhubTemplateFile" \
      --parameters eventhubName=$eventhubName \
          queryPipelineTopicName=$queryPipelineTopicName \
          ingestionPipelineTopicName=$ingestionPipelineTopicName \
      --debug
fi

# ------7. Azure Openai -------------------
# check if azure openai  exists

echo "Check if azure openai $azureOpenAIName exists...."

if az resource show \
    --name "$azureOpenAIName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.CognitiveServices/accounts" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $azureOpenAIName"
else
    echo "⚙️ Resource does not exist, creating it..."
    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$azureOpenAITemplateFile" \
      --parameters azureOpenAIName=$azureOpenAIName \
      --debug
fi

