// =============================================================================
// main.bicepparam — Production parameters for MMCT Container App deployment
//
// Usage:
//   az deployment group create \
//     -g DefaultResourceGroup-CCAN \
//     -f main.bicep \
//     -p main.bicepparam
//
// For secrets, set environment variables before deploying:
//   export NEO4J_PASSWORD='your-password'
// =============================================================================

using 'main.bicep'

// ---------------------------------------------------------------------------
// Infrastructure
// ---------------------------------------------------------------------------
param location = 'southeastasia'
param containerRegistryName = 'geckocontainerregistry'
param identityName = 'mmct-res-identity'
param imageName = 'mmct-lively-fastapi'
param imageTag = 'latest'

// ---------------------------------------------------------------------------
// Container Apps
// ---------------------------------------------------------------------------
param containerAppName = 'mmct-lively-fastapi-app'
param containerAppsEnvName = 'mmct-lively-fastapi-aca-env'

// ---------------------------------------------------------------------------
// Workload Profile — D4: 4 vCPU / 16 GiB per node
// ---------------------------------------------------------------------------
param workloadProfileName = 'dedicated-d4'
param workloadProfileType = 'D4'
param workloadProfileMinNodes = 1
param workloadProfileMaxNodes = 25

// ---------------------------------------------------------------------------
// Scaling — Target: thousands of parallel users
//   Min 3 replicas  ≈   600 concurrent users (warm pool)
//   Max 100 replicas ≈ 20,000 concurrent users (burst)
// ---------------------------------------------------------------------------
param scaleMinReplicas = 3
param scaleMaxReplicas = 100
param scaleConcurrentRequests = 50

// ---------------------------------------------------------------------------
// Container Resources (per replica)
// ---------------------------------------------------------------------------
param containerCpu = '4'
param containerMemory = '8Gi'

// ---------------------------------------------------------------------------
// LLM Provider (Azure OpenAI)
// ---------------------------------------------------------------------------
param llmEndpoint = 'https://geckooai.openai.azure.com/'
param llmDeploymentName = 'gpt-4.1-mini'
param llmModelName = 'gpt-4.1-mini'
param llmApiVersion = '2024-10-21'

// ---------------------------------------------------------------------------
// Embedding Provider (Azure OpenAI)
// ---------------------------------------------------------------------------
param embeddingServiceEndpoint = 'https://geckooai.openai.azure.com/'
param embeddingServiceDeploymentName = 'text-embedding-ada-002'
param embeddingServiceApiVersion = '2024-12-01-preview'
param embeddingServiceModelName = 'text-embedding-ada-002'

// ---------------------------------------------------------------------------
// Azure AI Search
// ---------------------------------------------------------------------------
param searchEndpoint = 'https://geckoaisearch-prod.search.windows.net'
param chapterIndexName = 'kv-dg-chapters'
param keyframesIndexName = 'kv-dg-keyframes'
param objectCollectionIndexName = 'kv-dg-objects'

// ---------------------------------------------------------------------------
// Azure Blob Storage
// ---------------------------------------------------------------------------
param storageAccountName = 'geckostorageaccount'

// ---------------------------------------------------------------------------
// Azure Speech Service
// ---------------------------------------------------------------------------
param speechServiceRegion = 'eastus'
param speechServiceResourceId = '/subscriptions/87f80cf2-3f33-456d-b999-8be499f65031/resourceGroups/DefaultResourceGroup-CCAN/providers/Microsoft.CognitiveServices/accounts/gecko-stt'

// ---------------------------------------------------------------------------
// Neo4j — Use environment variable for password:
//   export NEO4J_PASSWORD='StrongPass123!'
// ---------------------------------------------------------------------------
param neo4jUri = 'bolt://20.212.86.132:7687'
param neo4jUsername = 'neo4j'
param neo4jPassword = readEnvironmentVariable('NEO4J_PASSWORD', '')

// ---------------------------------------------------------------------------
// Event Hub (optional — leave empty to skip)
// ---------------------------------------------------------------------------
param eventHubHostname = ''
param ingestionEventHubName = ''
