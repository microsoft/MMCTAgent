#!/bin/bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$(realpath "$script_dir/../infra_config.yaml")"

get_yaml_value() {
    python -c "import yaml, sys; print(yaml.safe_load(sys.stdin.read())$1)" < "$CONFIG_FILE"
}

# Source the env vars using absolute path
source "$script_dir/00-setup-env-vars.sh"

# ------------------ DEPLOY RESOURCES -------------

check_and_deploy() {
    local resourceName=$1
    local templateFile=$2
    local type=$3
    local params=$4
    local yamlPath=$5

    if [[ -z "${!resourceName}" ]]; then
        echo "❌ Skipping $resourceName because the variable is empty."
        return
    fi

    if [[ "$(get_yaml_value "$yamlPath" | tr -d '[:space:]')" == "True" ]]; then
        echo "Checking if $resourceName '${!resourceName}' exists..."
        if az resource show \
            --name "${!resourceName}" \
            --resource-group "$resourceGroup" \
            --resource-type "$type" \
            --query "id" \
            --output tsv >/dev/null 2>&1; then
            echo "✅ Resource exists: Name = ${!resourceName}"
        else
            echo "⚙️ Resource does not exist, creating it..."
            az deployment group create \
                --resource-group "$resourceGroup" \
                --template-file "$templateFile" \
                --parameters $params \
                --debug
        fi
    else
        echo "=> Skipping $resourceName deployment..."
    fi
}

# 1. STORAGE ACCOUNT
check_and_deploy "storageAccountName" "$storageAccountTemplateFile" \
    "Microsoft.Storage/storageAccounts" "storageAccountName=$storageAccountName" "['deployInfra']['storageAccount']"

# 2. AI SEARCH SERVICE
check_and_deploy "aiSearchServiceName" "$aiSearchServiceTemplateFile" \
    "Microsoft.Search/searchServices" "aiSearchServiceName=$aiSearchServiceName" "['deployInfra']['aiSearchService']"

# 3. AZURE SPEECH SERVICE
check_and_deploy "azureSpeechServiceName" "$azureSpeechServiceTemplateFile" \
    "Microsoft.CognitiveServices/accounts" \
    "azureSpeechServiceName=$azureSpeechServiceName azureSpeechServiceRegion=$azureSpeechServiceRegion" \
    "['deployInfra']['azureSpeechService']"

# 4. CONTAINER REGISTRY
check_and_deploy "containerRegistryName" "$containerRegistryTemplateFile" \
    "Microsoft.ContainerRegistry/registries" \
    "containerRegistryName=$containerRegistryName" "['deployInfra']['containerRegistry']"

# 5. APP SERVICE PLAN - PREMIUM
check_and_deploy "aspPremiumName" "$aspPremiumTemplateFile" \
    "Microsoft.Web/serverfarms" "aspPremiumName=$aspPremiumName" "['deployInfra']['appServicePremiumPlan']"

# 6. APP SERVICE PLAN - BASIC
check_and_deploy "aspBasicName" "$aspBasicTemplateFile" \
    "Microsoft.Web/serverfarms" "aspBasicName=$aspBasicName" "['deployInfra']['appServiceBasicPlan']"

# 7. EVENT HUB
if [[ -z "$eventhubName" ]]; then
    echo "❌ Skipping event hub because the variable is empty."
else
    if [[ "$(get_yaml_value "['deployInfra']['eventHubService']" | tr -d '[:space:]')" == "True" ]]; then
        echo "Checking if event hub '$eventhubName' exists..."
        if az resource show \
            --name "$eventhubName" \
            --resource-group "$resourceGroup" \
            --resource-type "Microsoft.EventHub/namespaces" \
            --query "id" \
            --output tsv >/dev/null 2>&1; then
            echo "✅ Resource exists: Name = $eventhubName"
        else
            echo "⚙️ Resource does not exist, creating it..."
            az deployment group create \
                --resource-group "$resourceGroup" \
                --template-file "$eventhubTemplateFile" \
                --parameters eventhubName=$eventhubName \
                             queryPipelineTopicName=$queryPipelineTopicName \
                             ingestionPipelineTopicName=$ingestionPipelineTopicName \
                --debug
        fi
    else
        echo "=> Skipping event hub deployment..."
    fi
fi

# 8. AZURE OPENAI
check_and_deploy "azureOpenAIName" "$azureOpenAITemplateFile" \
    "Microsoft.CognitiveServices/accounts" "azureOpenAIName=$azureOpenAIName" "['deployInfra']['azureOpenAIService']"
