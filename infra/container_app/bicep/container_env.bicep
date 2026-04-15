// MMCTAgent — Container Apps Environment
//
// Creates the shared Azure infrastructure for running containerised workloads:
//   - Log Analytics Workspace  (receives all container stdout/stderr)
//   - Container Apps Environment (the runtime boundary for Container Apps)
//
// This template is intentionally thin. The Container App itself (image, env
// vars, replicas) is managed by infra/app/deploy.py via `az containerapp`
// CLI commands — this keeps .env secrets out of Bicep param files entirely.
//
// Deploy via:
//   python infra/app/deploy.py            (recommended)
//   az deployment group create …          (direct, step 4 of deploy.py)

targetScope = 'resourceGroup'

// ============================================================================
// Parameters
// ============================================================================

@description('Azure region for all resources')
param location string = 'eastus'

@description('Name of the Container Apps Environment')
param environmentName string = 'mmct-env'

@description('Name of the Log Analytics Workspace')
param logAnalyticsWorkspaceName string = 'mmct-logs'

// ============================================================================
// Log Analytics Workspace
// ============================================================================

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      searchVersion: 1
    }
  }
}

// ============================================================================
// Container Apps Environment
// ============================================================================

resource containerEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    zoneRedundant: false
  }
}

// ============================================================================
// Outputs
// ============================================================================

output environmentId   string = containerEnv.id
output environmentName string = containerEnv.name
output logAnalyticsId  string = logAnalytics.id