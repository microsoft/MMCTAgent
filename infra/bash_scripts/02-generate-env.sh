#!/bin/bash
set -e

# Load environment variable definitions
# Source the env vars using absolute path
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/00-setup-env-vars.sh"

# Output file
env_output_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../../ && pwd)/.env"

rm -f "$env_output_file"
touch "$env_output_file"

# Helper to check resource existence
resource_exists() {
    az resource show --resource-group "$resourceGroup" --name "$1" --resource-type "$2" &> /dev/null
}

# Write a key=value line
write_env() {
    echo "$1=$2" >> "$env_output_file"
}

write_comment_in_env() {
    echo -e "\n# $1" >> "$env_output_file"
}

# Start Writing
echo "# Generated .env File" >> "$env_output_file"

# ---------------- Blob Storage ----------------
write_comment_in_env "Blob Storage Container Configuration"
write_env "VIDEO_CONTAINER_NAME" "mmct-videocontainer"
write_env "FRAMES_CONTAINER_NAME" "mmct-framescontainer"
write_env "TIMESTAMPS_CONTAINER_NAME" "mmct-timestampscontainer"
write_env "TRANSCRIPT_CONTAINER_NAME" "mmct-transcriptcontainer"
write_env "AUDIO_CONTAINER_NAME" "mmct-audiocontainer"
write_env "SUMMARY_CONTAINER_NAME" "mmct-summary-n-transcript"
write_env "BLOB_DOWNLOAD_DIR" "media"
write_env "BLOB_MANAGED_IDENTITY" "True"

if [[ -n "$storageAccountName" ]] && resource_exists "$storageAccountName" "Microsoft.Storage/storageAccounts"; then
    blob_url="https://${storageAccountName}.blob.core.windows.net/"
    write_env "BLOB_ACCOUNT_URL" "$blob_url"
    write_env "BLOB_CONNECTION_STRING" ""
    write_env "BLOB_SAS_TOKEN" ""
fi

# ---------------- Azure Speech Service ----------------
write_comment_in_env "Azure Speech Service Configuration"
write_env "SPEECH_SERVICE_REGION" "$azureSpeechServiceRegion"
if resource_exists "$azureSpeechServiceName" "Microsoft.CognitiveServices/accounts"; then
    speech_resource_id=$(az resource show \
        --resource-group "$resourceGroup" \
        --name "$azureSpeechServiceName" \
        --resource-type "Microsoft.CognitiveServices/accounts" \
        --query "id" -o tsv)
    write_env "SPEECH_SERVICE_RESOURCE_ID" "$speech_resource_id"
fi

# ---------------- Azure CV ----------------
write_comment_in_env "Azure Computer Vision Configuration"
write_env "COMPUTER_VISION_ENDPOINT" ""
write_env "COMPUTER_VISION_API_VERSION" ""
write_env "MANAGED_IDENTITY" "True"
write_env "COMPUTER_VISION_KEY" ""

# ---------------- LLM Provider ----------------
write_comment_in_env "LLM Config"
write_env "LLM_PROVIDER" "azure"

# ---------------- Azure OpenAI ----------------
write_comment_in_env "Azure OpenAI Configuration"
if resource_exists "$azureOpenAIName" "Microsoft.CognitiveServices/accounts"; then
    openai_base_url="https://${azureOpenAIName}.openai.azure.com/"
    write_env "LLM_ENDPOINT" "$openai_base_url"
    write_env "LLM_DEPLOYMENT_NAME" "gpt-4o"
    write_env "LLM_MODEL_NAME" "gpt-4o"
    write_env "LLM_API_VERSION" "2024-08-01-preview"
    write_env "LLM_API_KEY" ""

    write_env "LLM_VISION_DEPLOYMENT_NAME" "gpt-4o"
    write_env "LLM_VISION_MODEL_NAME" "gpt-4o"
    write_env "LLM_VISION_API_VERSION" "2024-08-01-preview"

    write_env "EMBEDDING_SERVICE_ENDPOINT" "$openai_base_url"
    write_env "EMBEDDING_SERVICE_DEPLOYMENT_NAME" "text-embedding-ada-002"
    write_env "EMBEDDING_SERVICE_API_VERSION" "2023-05-15"
    write_env "EMBEDDING_SERVICE_MODEL_NAME" "text-embedding-ada-002"
    write_env "EMBEDDING_SERVICE_API_KEY" ""

    write_env "SPEECH_SERVICE_ENDPOINT" "$openai_base_url"
    write_env "SPEECH_SERVICE_DEPLOYMENT_NAME" "whisper"
    write_env "SPEECH_SERVICE_MODEL_NAME" "whisper"
    write_env "SPEECH_SERVICE_API_VERSION" "2024-06-01"
    write_env "SPEECH_SERVICE_KEY" ""
else
    write_env "MANAGED_IDENTITY" "False"
fi

# ---------------- OpenAI Config ----------------
write_comment_in_env "OPENAI Config"
write_env "OPENAI_MODEL_NAME" "gpt-4o"
write_env "OPENAI_VISION_MODEL_NAME" "gpt-4o-mini"
write_env "OPENAI_API_VERSION" "2024-08-01-preview"
write_env "OPENAI_VISION_API_VISION" "2024-08-01-preview"

write_env "OPENAI_EMBEDDING_MODEL_NAME" "text-embedding-ada-002"
write_env "OPENAI_EMBEDDING_API_VERSION" "2023-05-15"
write_env "OPENAI_SPEECH_SERVICE_MODEL_NAME" "whisper"
write_env "OPENAI_SPEECH_SERVICE_API_VERSION" "2024-06-01"

write_env "OPENAI_API_KEY" ""
write_env "OPENAI_SPEECH_SERVICE_KEY" ""
write_env "OPENAI_EMBEDDING_KEY" ""

# ---------------- Azure AI Search ----------------
write_comment_in_env "Azure AI Search"
if resource_exists "$aiSearchServiceName" "Microsoft.Search/searchServices"; then
    ai_search_url="https://${aiSearchServiceName}.search.windows.net"
    write_env "SEARCH_SERVICE_ENDPOINT" "$ai_search_url"
else
    write_env "SEARCH_SERVICE_ENDPOINT" ""
fi

write_env "SEARCH_SERVICE_KEY" ""

# ---------------- Event Hub ----------------
write_comment_in_env "Event Hub Configuration"
if resource_exists "$eventhubName" "Microsoft.EventHub/namespaces"; then
    hostname="${eventhubName}.servicebus.windows.net"
    write_env "EVENT_HUB_HOSTNAME" "$hostname"
else
    write_env "EVENT_HUB_HOSTNAME" ""
fi
write_env "QUERY_EVENT_HUB_NAME" "$queryPipelineTopicName"
write_env "INGESTION_EVENT_HUB_NAME" "$ingestionPipelineTopicName"

# ---------------- Final Message ----------------
echo -e "\n✅ Generated: $env_output_file"