// =============================================================================
// main.bicep — MMCT FastAPI Server on Azure Container Apps (High-Scale)
//
// Deploys with auto-scaling for thousands of parallel users:
//   - Dedicated D4 workload profile (4 vCPU / 16 GiB per node)
//   - Min 3 replicas (warm pool, ~600 concurrent users instantly)
//   - Max 100 replicas (burst to ~20,000 concurrent users)
//   - HTTP-based scaling: +1 replica per 50 concurrent requests
//
// Usage:
//   az deployment group create \
//     -g DefaultResourceGroup-CCAN \
//     -f main.bicep \
//     -p main.bicepparam
// =============================================================================

targetScope = 'resourceGroup'

// ---------------------------------------------------------------------------
// Parameters — Infrastructure
// ---------------------------------------------------------------------------

@description('Azure region for the Container Apps environment')
param location string

@description('Name of the existing Azure Container Registry')
param containerRegistryName string

@description('Name of the existing user-assigned managed identity')
param identityName string

@description('Resource group of the managed identity (if different from deployment RG)')
param identityResourceGroup string = resourceGroup().name

@description('Docker image name (without registry prefix or tag)')
param imageName string = 'mmct-lively-fastapi'

@description('Docker image tag')
param imageTag string = 'latest'

@description('Container App name')
param containerAppName string = 'mmct-main-app'

@description('Container Apps Environment name')
param containerAppsEnvName string = 'mmct-aca-env'

// ---------------------------------------------------------------------------
// Parameters — Workload Profile
// ---------------------------------------------------------------------------

@description('Workload profile name')
param workloadProfileName string = 'dedicated-d4'

@description('Workload profile type (D4 = 4 vCPU / 16 GiB)')
@allowed(['D4', 'D8', 'D16', 'D32', 'E4', 'E8', 'E16', 'E32'])
param workloadProfileType string = 'D4'

@minValue(1)
@maxValue(50)
@description('Minimum dedicated nodes in the workload profile')
param workloadProfileMinNodes int = 1

@minValue(1)
@maxValue(50)
@description('Maximum dedicated nodes in the workload profile')
param workloadProfileMaxNodes int = 25

// ---------------------------------------------------------------------------
// Parameters — Container Resources & Scaling
// ---------------------------------------------------------------------------

@description('CPU cores per replica')
param containerCpu string = '4'

@description('Memory per replica (e.g., 8Gi)')
param containerMemory string = '8Gi'

@minValue(0)
@maxValue(300)
@description('Minimum replicas (warm pool). 3 ≈ 600 concurrent users.')
param scaleMinReplicas int = 3

@minValue(1)
@maxValue(300)
@description('Maximum replicas (burst ceiling). 100 ≈ 20,000 concurrent users.')
param scaleMaxReplicas int = 100

@minValue(1)
@description('New replica added per N concurrent HTTP requests')
param scaleConcurrentRequests int = 50

// ---------------------------------------------------------------------------
// Parameters — LLM Provider (Azure OpenAI)
// ---------------------------------------------------------------------------

@description('Azure OpenAI endpoint URL')
param llmEndpoint string

@description('LLM deployment name')
param llmDeploymentName string = 'gpt-4o'

@description('LLM model name')
param llmModelName string = 'gpt-4o'

@description('LLM API version')
param llmApiVersion string = '2025-01-01-preview'

// ---------------------------------------------------------------------------
// Parameters — Embedding Provider
// ---------------------------------------------------------------------------

@description('Embedding service endpoint URL')
param embeddingServiceEndpoint string

@description('Embedding deployment name')
param embeddingServiceDeploymentName string = 'text-embedding-ada-002'

@description('Embedding API version')
param embeddingServiceApiVersion string = '2024-12-01-preview'

@description('Embedding model name')
param embeddingServiceModelName string = 'text-embedding-ada-002'

// ---------------------------------------------------------------------------
// Parameters — Azure AI Search
// ---------------------------------------------------------------------------

@description('Azure AI Search endpoint URL')
param searchEndpoint string

@description('Chapter index name')
param chapterIndexName string

@description('Keyframes index name')
param keyframesIndexName string

@description('Object collection index name')
param objectCollectionIndexName string

// ---------------------------------------------------------------------------
// Parameters — Azure Blob Storage
// ---------------------------------------------------------------------------

@description('Storage account name')
param storageAccountName string

// ---------------------------------------------------------------------------
// Parameters — Azure Speech Service
// ---------------------------------------------------------------------------

@description('Speech service region')
param speechServiceRegion string = 'eastus'

@description('Speech service resource ID (full ARM path)')
param speechServiceResourceId string

// ---------------------------------------------------------------------------
// Parameters — Neo4j
// ---------------------------------------------------------------------------

@description('Neo4j bolt URI')
param neo4jUri string = 'bolt://localhost:7687'

@description('Neo4j username')
param neo4jUsername string = 'neo4j'

@secure()
@description('Neo4j password')
param neo4jPassword string

// ---------------------------------------------------------------------------
// Parameters — Event Hub (optional)
// ---------------------------------------------------------------------------

@description('Event Hub hostname (leave empty to skip)')
param eventHubHostname string = ''

@description('Ingestion Event Hub name (leave empty to skip)')
param ingestionEventHubName string = ''

// ---------------------------------------------------------------------------
// Existing Resources — resolved dynamically (no hardcoded IDs)
// ---------------------------------------------------------------------------

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: identityName
  scope: resourceGroup(identityResourceGroup)
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
}

// ---------------------------------------------------------------------------
// Container Apps Environment with Dedicated Workload Profile
// ---------------------------------------------------------------------------

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppsEnvName
  location: location
  properties: {
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
      {
        name: workloadProfileName
        workloadProfileType: workloadProfileType
        minimumCount: workloadProfileMinNodes
        maximumCount: workloadProfileMaxNodes
      }
    ]
  }
}

// ---------------------------------------------------------------------------
// Container App — MMCT FastAPI Server
// ---------------------------------------------------------------------------

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    workloadProfileName: workloadProfileName
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
      registries: [
        {
          server: '${acr.name}.azurecr.io'
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          image: '${acr.name}.azurecr.io/${imageName}:${imageTag}'
          name: 'mmct-main-app'
          resources: {
            cpu: json(containerCpu)
            memory: containerMemory
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              periodSeconds: 10
              failureThreshold: 3
              initialDelaySeconds: 10
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              periodSeconds: 5
              failureThreshold: 3
              initialDelaySeconds: 5
            }
            {
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: 8000
              }
              periodSeconds: 5
              failureThreshold: 10
              initialDelaySeconds: 3
            }
          ]
          env: [
            // --- Managed Identity ---
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
            { name: 'MANAGED_IDENTITY_CLIENT_ID', value: identity.properties.clientId }
            { name: 'MANAGED_IDENTITY', value: 'true' }

            // --- LLM Provider (Azure OpenAI) ---
            { name: 'LLM_PROVIDER', value: 'azure' }
            { name: 'LLM_USE_MANAGED_IDENTITY', value: 'true' }
            { name: 'LLM_TIMEOUT', value: '200' }
            { name: 'LLM_MAX_RETRIES', value: '2' }
            { name: 'LLM_TEMPERATURE', value: '0.0' }
            { name: 'LLM_ENDPOINT', value: llmEndpoint }
            { name: 'LLM_DEPLOYMENT_NAME', value: llmDeploymentName }
            { name: 'LLM_MODEL_NAME', value: llmModelName }
            { name: 'LLM_API_VERSION', value: llmApiVersion }
            { name: 'LLM_API_KEY', value: '' }
            { name: 'LLM_VISION_DEPLOYMENT_NAME', value: llmDeploymentName }
            { name: 'LLM_VISION_API_VERSION', value: llmApiVersion }

            // --- Embedding Provider ---
            { name: 'EMBEDDING_PROVIDER', value: 'azure' }
            { name: 'EMBEDDING_USE_MANAGED_IDENTITY', value: 'true' }
            { name: 'EMBEDDING_TIMEOUT', value: '200' }
            { name: 'EMBEDDING_SERVICE_ENDPOINT', value: embeddingServiceEndpoint }
            { name: 'EMBEDDING_SERVICE_DEPLOYMENT_NAME', value: embeddingServiceDeploymentName }
            { name: 'EMBEDDING_SERVICE_API_VERSION', value: embeddingServiceApiVersion }
            { name: 'EMBEDDING_SERVICE_MODEL_NAME', value: embeddingServiceModelName }
            { name: 'EMBEDDING_SERVICE_API_KEY', value: '' }

            // --- Azure AI Search ---
            { name: 'SEARCH_PROVIDER', value: 'azure_ai_search' }
            { name: 'SEARCH_USE_MANAGED_IDENTITY', value: 'true' }
            { name: 'SEARCH_TIMEOUT', value: '30' }
            { name: 'SEARCH_ENDPOINT', value: searchEndpoint }
            { name: 'SEARCH_API_KEY', value: '' }
            { name: 'CHAPTER_INDEX_NAME', value: chapterIndexName }
            { name: 'KEYFRAMES_INDEX_NAME', value: keyframesIndexName }
            { name: 'OBJECT_COLLECTION_INDEX_NAME', value: objectCollectionIndexName }

            // --- Storage (Azure Blob) ---
            { name: 'STORAGE_PROVIDER', value: 'azure' }
            { name: 'STORAGE_USE_MANAGED_IDENTITY', value: 'true' }
            { name: 'STORAGE_ACCOUNT_NAME', value: storageAccountName }
            { name: 'BLOB_ACCOUNT_URL', value: 'https://${storageAccountName}.blob.${az.environment().suffixes.storage}' }
            { name: 'BLOB_MANAGED_IDENTITY', value: 'true' }
            { name: 'VIDEO_CONTAINER_NAME', value: 'mmct-videocontainer' }
            { name: 'FRAMES_CONTAINER_NAME', value: 'mmct-framescontainer' }
            { name: 'TIMESTAMPS_CONTAINER_NAME', value: 'mmct-timestampscontainer' }
            { name: 'TRANSCRIPT_CONTAINER_NAME', value: 'mmct-transcriptcontainer' }
            { name: 'AUDIO_CONTAINER_NAME', value: 'mmct-audiocontainer' }
            { name: 'VIDEO_DESCRIPTION_CONTAINER_NAME', value: 'mmct-summary-n-transcript' }
            { name: 'KEYFRAME_CONTAINER_NAME', value: 'keyframes' }
            { name: 'BLOB_DOWNLOAD_DIR', value: 'media' }

            // --- Speech Service ---
            { name: 'TRANSCRIPTION_PROVIDER', value: 'azure' }
            { name: 'SPEECH_USE_MANAGED_IDENTITY', value: 'true' }
            { name: 'SPEECH_TIMEOUT', value: '200' }
            { name: 'SPEECH_SERVICE_REGION', value: speechServiceRegion }
            { name: 'SPEECH_SERVICE_RESOURCE_ID', value: speechServiceResourceId }

            // --- Neo4j (V4 Graph Backend) ---
            { name: 'NEO4J_URI', value: neo4jUri }
            { name: 'NEO4J_USERNAME', value: neo4jUsername }
            { name: 'NEO4J_PASSWORD', value: neo4jPassword }

            // --- Vision Provider ---
            { name: 'VISION_PROVIDER', value: 'azure' }

            // --- Event Hub ---
            { name: 'EVENT_HUB_HOSTNAME', value: eventHubHostname }
            { name: 'INGESTION_EVENT_HUB_NAME', value: ingestionEventHubName }

            // --- Application ---
            { name: 'APP_NAME', value: 'MMCT Agent' }
            { name: 'APP_VERSION', value: '1.0.0' }
            { name: 'ENVIRONMENT', value: 'production' }
            { name: 'DEBUG', value: 'false' }
            { name: 'LOG_LEVEL', value: 'INFO' }
            { name: 'LOG_ENABLE_JSON', value: 'false' }
            { name: 'LOG_ENABLE_FILE', value: 'false' }
            { name: 'ENABLE_SECRETS_MANAGER', value: 'false' }
          ]
        }
      ]
      scale: {
        minReplicas: scaleMinReplicas
        maxReplicas: scaleMaxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: string(scaleConcurrentRequests)
              }
            }
          }
        ]
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

@description('Container App FQDN')
output fqdn string = containerApp.properties.configuration.ingress.fqdn

@description('Container App URL')
output url string = 'https://${containerApp.properties.configuration.ingress.fqdn}'

@description('Container App resource ID')
output resourceId string = containerApp.id

@description('Container Apps Environment resource ID')
output environmentId string = environment.id
