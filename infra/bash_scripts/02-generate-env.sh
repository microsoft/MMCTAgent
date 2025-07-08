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

# ---------------- Application Settings ----------------
write_comment_in_env "Application Settings"
write_env "APP_NAME" "MMCT Agent"
write_env "APP_VERSION" "1.0.0"
write_env "ENVIRONMENT" "production"
write_env "DEBUG" "false"

# ---------------- LLM Provider Configuration ----------------
write_comment_in_env "LLM Provider Configuration"
write_env "LLM_PROVIDER" "azure"
write_env "LLM_USE_MANAGED_IDENTITY" "true"
write_env "LLM_TIMEOUT" "200"
write_env "LLM_MAX_RETRIES" "2"
write_env "LLM_TEMPERATURE" "0.0"

if resource_exists "$azureOpenAIName" "Microsoft.CognitiveServices/accounts"; then
    openai_base_url="https://${azureOpenAIName}.openai.azure.com/"
    write_env "LLM_ENDPOINT" "$openai_base_url"
    write_env "LLM_DEPLOYMENT_NAME" "gpt-4o"
    write_env "LLM_MODEL_NAME" "gpt-4o"
    write_env "LLM_API_VERSION" "2024-08-01-preview"
    write_env "LLM_API_KEY" ""
    write_env "LLM_VISION_DEPLOYMENT_NAME" "gpt-4o"
    write_env "LLM_VISION_API_VERSION" "2024-08-01-preview"
else
    write_env "LLM_ENDPOINT" ""
    write_env "LLM_DEPLOYMENT_NAME" "gpt-4o"
    write_env "LLM_MODEL_NAME" "gpt-4o"
    write_env "LLM_API_VERSION" "2024-08-01-preview"
    write_env "LLM_API_KEY" ""
    write_env "LLM_VISION_DEPLOYMENT_NAME" "gpt-4o"
    write_env "LLM_VISION_API_VERSION" "2024-08-01-preview"
fi

# ---------------- Search Provider Configuration ----------------
write_comment_in_env "Search Provider Configuration"
write_env "SEARCH_PROVIDER" "azure_ai_search"
write_env "SEARCH_USE_MANAGED_IDENTITY" "true"
write_env "SEARCH_INDEX_NAME" "default"
write_env "SEARCH_TIMEOUT" "30"

if resource_exists "$aiSearchServiceName" "Microsoft.Search/searchServices"; then
    ai_search_url="https://${aiSearchServiceName}.search.windows.net"
    write_env "SEARCH_ENDPOINT" "$ai_search_url"
else
    write_env "SEARCH_ENDPOINT" ""
fi
write_env "SEARCH_API_KEY" ""

# ---------------- Embedding Provider Configuration ----------------
write_comment_in_env "Embedding Provider Configuration"
write_env "EMBEDDING_PROVIDER" "azure"
write_env "EMBEDDING_USE_MANAGED_IDENTITY" "true"
write_env "EMBEDDING_TIMEOUT" "200"

if resource_exists "$azureOpenAIName" "Microsoft.CognitiveServices/accounts"; then
    write_env "EMBEDDING_SERVICE_ENDPOINT" "$openai_base_url"
    write_env "EMBEDDING_SERVICE_DEPLOYMENT_NAME" "text-embedding-ada-002"
    write_env "EMBEDDING_SERVICE_API_VERSION" "2024-08-01-preview"
    write_env "EMBEDDING_SERVICE_MODEL_NAME" "text-embedding-ada-002"
    write_env "EMBEDDING_SERVICE_API_KEY" ""
else
    write_env "EMBEDDING_SERVICE_ENDPOINT" ""
    write_env "EMBEDDING_SERVICE_DEPLOYMENT_NAME" "text-embedding-ada-002"
    write_env "EMBEDDING_SERVICE_API_VERSION" "2024-08-01-preview"
    write_env "EMBEDDING_SERVICE_MODEL_NAME" "text-embedding-ada-002"
    write_env "EMBEDDING_SERVICE_API_KEY" ""
fi

# ---------------- Transcription Provider Configuration ----------------
write_comment_in_env "Transcription Provider Configuration"
write_env "TRANSCRIPTION_PROVIDER" "azure"
write_env "SPEECH_USE_MANAGED_IDENTITY" "true"
write_env "SPEECH_TIMEOUT" "200"
write_env "SPEECH_SERVICE_REGION" "$azureSpeechServiceRegion"

if resource_exists "$azureSpeechServiceName" "Microsoft.CognitiveServices/accounts"; then
    speech_resource_id=$(az resource show \
        --resource-group "$resourceGroup" \
        --name "$azureSpeechServiceName" \
        --resource-type "Microsoft.CognitiveServices/accounts" \
        --query "id" -o tsv)
    write_env "SPEECH_SERVICE_RESOURCE_ID" "$speech_resource_id"
    
    # For Azure Speech Service (not OpenAI Whisper via Azure OpenAI)
    speech_endpoint="https://${azureSpeechServiceRegion}.api.cognitive.microsoft.com/"
    write_env "SPEECH_SERVICE_ENDPOINT" "$speech_endpoint"
    write_env "SPEECH_SERVICE_DEPLOYMENT_NAME" "whisper"
    write_env "SPEECH_SERVICE_API_VERSION" "2024-08-01-preview"
    write_env "SPEECH_SERVICE_KEY" ""
else
    write_env "SPEECH_SERVICE_RESOURCE_ID" ""
    write_env "SPEECH_SERVICE_ENDPOINT" ""
    write_env "SPEECH_SERVICE_DEPLOYMENT_NAME" "whisper"
    write_env "SPEECH_SERVICE_API_VERSION" "2024-08-01-preview"
    write_env "SPEECH_SERVICE_KEY" ""
fi

# ---------------- Vision Provider Configuration ----------------
write_comment_in_env "Vision Provider Configuration"
write_env "VISION_PROVIDER" "azure"

# ---------------- Storage Configuration ----------------
write_comment_in_env "Storage Configuration"
write_env "STORAGE_PROVIDER" "azure_blob"
write_env "STORAGE_USE_MANAGED_IDENTITY" "true"
write_env "STORAGE_CONTAINER_NAME" "default"

if [[ -n "$storageAccountName" ]] && resource_exists "$storageAccountName" "Microsoft.Storage/storageAccounts"; then
    write_env "STORAGE_ACCOUNT_NAME" "$storageAccountName"
    blob_url="https://${storageAccountName}.blob.core.windows.net/"
    write_env "STORAGE_CONNECTION_STRING" ""
    
    # Legacy blob storage variables (for backward compatibility)
    write_env "BLOB_ACCOUNT_URL" "$blob_url"
    write_env "BLOB_CONNECTION_STRING" ""
    write_env "BLOB_SAS_TOKEN" ""
else
    write_env "STORAGE_ACCOUNT_NAME" ""
    write_env "STORAGE_CONNECTION_STRING" ""
    write_env "BLOB_ACCOUNT_URL" ""
    write_env "BLOB_CONNECTION_STRING" ""
    write_env "BLOB_SAS_TOKEN" ""
fi

# ---------------- Storage Container Configuration ----------------
write_comment_in_env "Storage Container Configuration"
write_env "VIDEO_CONTAINER_NAME" "mmct-videocontainer"
write_env "FRAMES_CONTAINER_NAME" "mmct-framescontainer"
write_env "TIMESTAMPS_CONTAINER_NAME" "mmct-timestampscontainer"
write_env "TRANSCRIPT_CONTAINER_NAME" "mmct-transcriptcontainer"
write_env "AUDIO_CONTAINER_NAME" "mmct-audiocontainer"
write_env "SUMMARY_CONTAINER_NAME" "mmct-summary-n-transcript"
write_env "BLOB_DOWNLOAD_DIR" "media"
write_env "BLOB_MANAGED_IDENTITY" "true"

# ---------------- Azure Computer Vision Configuration ----------------
write_comment_in_env "Azure Computer Vision Configuration"
write_env "COMPUTER_VISION_ENDPOINT" ""
write_env "COMPUTER_VISION_API_VERSION" "2024-02-01"
write_env "COMPUTER_VISION_KEY" ""

# ---------------- Security Configuration ----------------
write_comment_in_env "Security Configuration"
write_env "KEYVAULT_URL" ""
write_env "ENABLE_SECRETS_MANAGER" "false"
write_env "MANAGED_IDENTITY_CLIENT_ID" ""
write_env "MANAGED_IDENTITY" "true"

# ---------------- Logging Configuration ----------------
write_comment_in_env "Logging Configuration"
write_env "LOG_LEVEL" "INFO"
write_env "LOG_FILE" "logs/mmct-prod.log"
write_env "LOG_ENABLE_JSON" "true"
write_env "LOG_ENABLE_FILE" "true"
write_env "LOG_MAX_FILE_SIZE" "10 MB"
write_env "LOG_RETENTION_DAYS" "30"

# ---------------- Feature Flags ----------------
write_comment_in_env "Feature Flags"
write_env "ENABLE_CACHING" "true"
write_env "ENABLE_METRICS" "true"
write_env "ENABLE_TRACING" "true"
write_env "ENABLE_RATE_LIMITING" "true"

# ---------------- OpenAI Configuration (for hybrid setups) ----------------
write_comment_in_env "OpenAI Configuration (for hybrid setups)"
write_env "OPENAI_MODEL_NAME" "gpt-4o"
write_env "OPENAI_VISION_MODEL_NAME" "gpt-4o-mini"
write_env "OPENAI_API_VERSION" "2024-08-01-preview"
write_env "OPENAI_VISION_API_VERSION" "2024-08-01-preview"
write_env "OPENAI_EMBEDDING_MODEL_NAME" "text-embedding-ada-002"
write_env "OPENAI_EMBEDDING_API_VERSION" "2023-05-15"
write_env "OPENAI_SPEECH_SERVICE_MODEL_NAME" "whisper-1"
write_env "OPENAI_SPEECH_SERVICE_API_VERSION" "2024-06-01"
write_env "OPENAI_WHISPER_MODEL" "whisper-1"
write_env "OPENAI_API_KEY" ""
write_env "OPENAI_SPEECH_SERVICE_KEY" ""
write_env "OPENAI_EMBEDDING_KEY" ""

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

# ---------------- Legacy Environment Variables (for backward compatibility) ----------------
write_comment_in_env "Legacy Environment Variables (for backward compatibility)"
if resource_exists "$azureOpenAIName" "Microsoft.CognitiveServices/accounts"; then
    write_env "AZURE_OPENAI_ENDPOINT" "$openai_base_url"
    write_env "AZURE_OPENAI_DEPLOYMENT" "gpt-4o"
    write_env "AZURE_OPENAI_API_KEY" ""
fi

if resource_exists "$aiSearchServiceName" "Microsoft.Search/searchServices"; then
    write_env "AZURE_AI_SEARCH_ENDPOINT" "$ai_search_url"
    write_env "AZURE_AI_SEARCH_KEY" ""
fi

# Legacy variables for backward compatibility
write_env "SEARCH_SERVICE_ENDPOINT" "$ai_search_url"
write_env "SEARCH_SERVICE_KEY" ""

# ---------------- Final Message ----------------
echo -e "\n✅ Generated: $env_output_file"
echo "📝 Configuration includes:"
echo "   - New provider system variables"
echo "   - Correct Azure OpenAI deployment names from ARM template"
echo "   - Security and logging configuration"
echo "   - Feature flags and performance settings"
echo "   - Legacy variables for backward compatibility"
echo "⚠️  Please review and update API keys, endpoints, and security settings as needed"