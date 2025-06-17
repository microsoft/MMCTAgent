
# --------------  SET VARIABLES --------------------
set -e
# set the name of the resource group
resourceGroup="test-arm-rg"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="$script_dir/../arm_templates"

# set the name and path of the resources

## 1. set the storage account name
storageAccountName="ostsa"
storageAccountTemplateFile="$base_dir/storage_account.json"

## 2. set the ai searh service name
aiSearchServiceName="ostais"
aiSearchServiceTemplateFile="$base_dir/azure_ai_search.json"

## 3. set the azure speech service name
azureSpeechServiceName="oststt"
azureSpeechServiceRegion="centralindia"
azureSpeechServiceTemplateFile="$base_dir/azure_speech_service.json"

## 4. set the container registry name
containerRegistryName="ostacr"
containerRegistryTemplateFile="$base_dir/container_registry.json"

## 5. set the app service PLAN name
aspPremiumName="ostaspp"
aspPremiumTemplateFile="$base_dir/app_service_plan_premium.json"

aspBasicName="ostaspb"
aspBasicTemplateFile="$base_dir/app_service_plan_basic.json"

## 6. set the azure event hub name and topic name
eventhubName="ostevhub"
queryPipelineTopicName="query-eventhub"
ingestionPipelineTopicName="ingestion-eventhub"
eventhubTemplateFile="$base_dir/azure_event_hub.json"

## 7. Azure Openai
azureOpenAIName="ostazoai"
azureOpenAITemplateFile="$base_dir/azure_openai.json"

## 8. Managed Identity Name
identityName="ostmidentity"

## 9. Docker Images, App service and Container Apps Name
imageTag="1.0"

# name the app service variables
ingestionProducerAppName="ostingestproducer"
ingestionProducerTemplateFile="$base_dir/ingestion_app_service.json"
ingestionProducerImageAndTag="${containerRegistryName}.azurecr.io/ost-ingestion-producer:${imageTag}"

queryProducerAppName="ostqueryproducer"
queryProducerTemplateFile="$base_dir/query_app_service.json"
queryProducerImageAndTag="${containerRegistryName}.azurecr.io/ost-query-producer:${imageTag}"

# name the container apps environment name
containerAppsEnvName="oscontappenv"

containerRegistry="$containerRegistryName.azurecr.io"
queryContainerImage="$containerRegistryName.azurecr.io/ost-query-consumer:$imageTag"
querycontaineAppName="ostqueryconsumer"
ingestioncontainerAppName="ostingestionconsumer"
ingestionContainerImage="$containerRegistryName.azurecr.io/ost-ingestion-consumer:$imageTag"