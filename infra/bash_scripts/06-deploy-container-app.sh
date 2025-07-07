#!/bin/bash
export MSYS_NO_PATHCONV=1

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the env vars using absolute path
source "$script_dir/00-setup-env-vars.sh"

## Container Apps Config
CPU="2"
MEMORY="4Gi"
MIN_REPLICAS=3
MAX_REPLICAS=10

echo "Fetching user-assigned identity resource ID…"
USERID=$(az identity show \
  --resource-group "$resourceGroup" \
  --name "$identityName" \
   --query id -o tsv | tr -d '\r')

echo "Identity ID: '$USERID'"

if [[ -z "$USERID" ]]; then
  echo "❌ Identity '$identityName' not found in RG '$resourceGroup'" >&2
  exit 1
fi

echo "Ensuring Container Apps environment '$containerAppsEnvName' exists…"
az containerapp env show \
  --name "$containerAppsEnvName" \
  --resource-group "$resourceGroup" &>/dev/null \
|| az containerapp env create \
     --name "$containerAppsEnvName" \
     --resource-group "$resourceGroup" \
     --location "Central India"

clientIdOfIdentity=$(az identity show \
  --resource-group $resourceGroup \
  --name $identityName \
  --query clientId \
  --output tsv)

echo "Client Id: $clientIdOfIdentity"

speechResourceId=$(az resource show \
    --resource-group "$resourceGroup" \
    --name "$azureSpeechServiceName" \
    --resource-type "Microsoft.CognitiveServices/accounts" \
    --query "id" \
    --output tsv)

echo "Speech Resource Id: $speechResourceId"

echo "Deploying Container App '$containerAppName' (with registry identity)…"

if az resource show \
    --name "$containerAppName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.App/containerApps" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $containerAppName"
else 
    az containerapp create \
  --name "$containerAppName" \
  --resource-group "$resourceGroup" \
  --environment "$containerAppsEnvName" \
  --image "$containerAppImageAndTag" \
  --registry-server "$containerRegistry" \
  --registry-identity  "$USERID" \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --min-replicas "$MIN_REPLICAS" \
  --max-replicas "$MAX_REPLICAS"  \
  --env-vars AZURE_CLIENT_ID="$clientIdOfIdentity" \
        VIDEO_CONTAINER_NAME="mmct-videocontainer" \
        FRAMES_CONTAINER_NAME="mmct-framescontainer" \
        TIMESTAMPS_CONTAINER_NAME="mmct-timestampscontainer" \
        TRANSCRIPT_CONTAINER_NAME="mmct-transcriptcontainer" \
        AUDIO_CONTAINER_NAME="mmct-audiocontainer" \
        SUMMARY_CONTAINER_NAME="mmct-summary-n-transcript" \
        BLOB_DOWNLOAD_DIR="media" \
        BLOB_MANAGED_IDENTITY="True" \
        BLOB_ACCOUNT_URL="https://$storageAccountName.blob.core.windows.net" \
        SPEECH_SERVICE_REGION="$azureSpeechServiceRegion" \
        SPEECH_SERVICE_RESOURCE_ID="$speechResourceId" \
        COMPUTER_VISION_ENDPOINT="" \
        COMPUTER_VISION_API_VERSION="" \
        MANAGED_IDENTITY="True" \
        LLM_PROVIDER="azure" \
        LLM_ENDPOINT="https://$azureOpenAIName.openai.azure.com/" \
        LLM_DEPLOYMENT_NAME="gpt-4o" \
        LLM_MODEL_NAME="gpt-4o" \
        LLM_API_VERSION="2024-08-01-preview" \
        LLM_VISION_DEPLOYMENT_NAME="gpt-4o" \
        LLM_VISION_MODEL_NAME="gpt-4o" \
        LLM_VISION_API_VERSION="2024-08-01-preview" \
        EMBEDDING_SERVICE_ENDPOINT="https://$azureOpenAIName.openai.azure.com/" \
        EMBEDDING_SERVICE_DEPLOYMENT_NAME="text-embedding-ada-002" \
        EMBEDDING_SERVICE_API_VERSION="2023-05-15" \
        EMBEDDING_SERVICE_MODEL_NAME="text-embedding-ada-002" \
        SPEECH_SERVICE_ENDPOINT="https://$azureOpenAIName.openai.azure.com" \
        SPEECH_SERVICE_DEPLOYMENT_NAME="whisper" \
        SPEECH_SERVICE_MODEL_NAME="whisper" \
        SPEECH_SERVICE_API_VERSION="2024-06-01" \
        OPENAI_MODEL_NAME="gpt-4o" \
        OPENAI_VISION_MODEL_NAME="gpt-4o-mini" \
        OPENAI_API_VERSION="2024-08-01-preview" \
        OPENAI_VISION_API_VISION="2024-08-01-preview" \
        OPENAI_EMBEDDING_MODEL_NAME="text-embedding-ada-002" \
        OPENAI_EMBEDDING_API_VERSION="2023-05-15" \
        OPENAI_SPEECH_SERVICE_MODEL_NAME="whisper" \
        OPENAI_SPEECH_SERVICE_API_VERSION="2024-06-01" \
        SEARCH_SERVICE_ENDPOINT="https://$aiSearchServiceName.search.windows.net" \
        EVENT_HUB_HOSTNAME="$eventhubName.servicebus.windows.net" \
        QUERY_EVENT_HUB_NAME="query-eventhub" \
        INGESTION_EVENT_HUB_NAME="ingestion-eventhub" \
  --output table 
fi

echo "$USERID"

echo "Assigning your user-managed identity to the Container App…"
az containerapp identity assign \
  --name "$containerAppName" \
  --resource-group "$resourceGroup" \
  --user-assigned "$USERID" \

echo "✅ Deployment complete."
echo "Check status with:"
echo "  az containerapp show -n $containerAppName -g $resourceGroup --query \"{Status:properties.provisioningState, URL:properties.configuration.ingress.fqdn}\" -o table"