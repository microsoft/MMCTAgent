#!/bin/bash
set -e
export MSYS_NO_PATHCONV=1

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$(realpath "$script_dir/../infra_config.yaml")"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
else
    echo "✅ Config file found!"
fi

get_yaml_value() {
    python -c "import yaml, sys; print(yaml.safe_load(sys.stdin.read())$1)" < "$CONFIG_FILE"
}

# Check if Windows is true
is_windows="$(get_yaml_value "['operatingSystem']['windows']")"

convert_path_if_windows() {
    local input_path="$1"
    if [[ "$is_windows" == "True" ]]; then
        cygpath -w "$input_path"
    else
        echo "$input_path"
    fi
}

# --------------  SET VARIABLES --------------------

resourceGroup="test-arm-rg"
base_dir="$script_dir/../arm_templates"

# Templates and Resources
storageAccountName="ossa"
storageAccountTemplateFile="$(convert_path_if_windows "$base_dir/storage_account.json")"

aiSearchServiceName="osais"
aiSearchServiceTemplateFile="$(convert_path_if_windows "$base_dir/azure_ai_search.json")"

azureSpeechServiceName="osstt"
azureSpeechServiceRegion="centralindia"
azureSpeechServiceTemplateFile="$(convert_path_if_windows "$base_dir/azure_speech_service.json")"

containerRegistryName="osacr"
containerRegistryTemplateFile="$(convert_path_if_windows "$base_dir/container_registry.json")"

aspPremiumName="osaspp"
aspPremiumTemplateFile="$(convert_path_if_windows "$base_dir/app_service_plan_premium.json")"

aspBasicName="osaspb"
aspBasicTemplateFile="$(convert_path_if_windows "$base_dir/app_service_plan_basic.json")"

eventhubName="oseventhub"
queryPipelineTopicName="query-eventhub"
ingestionPipelineTopicName="ingestion-eventhub"
eventhubTemplateFile="$(convert_path_if_windows "$base_dir/azure_event_hub.json")"

azureOpenAIName="osazoai"
azureOpenAITemplateFile="$(convert_path_if_windows "$base_dir/azure_openai.json")"

identityName="osmidentity"

imageTag="1.0"
baseImage="${containerRegistryName}.azurecr.io/osbase:${imageTag}"
mainAppServiceName="osmainappservice"
mainAppTemplateFile="$(convert_path_if_windows "$base_dir/main_app_service.json")"
mainAppImageandTag="${containerRegistryName}.azurecr.io/main-app:${imageTag}"

containerAppName="osingestioncons"
containerAppImageAndTag="${containerRegistryName}.azurecr.io/ingestion-consumer:${imageTag}"

containerAppsEnvName="oscontappenv"
containerRegistry="${containerRegistryName}.azurecr.io"
