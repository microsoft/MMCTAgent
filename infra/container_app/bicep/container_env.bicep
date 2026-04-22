// MMCTAgent — Container Apps Environment with VNet + NAT Gateway
//
// Creates the shared Azure infrastructure for running containerised workloads:
//   - Virtual Network + Subnet (delegated to Container Apps)
//   - NAT Gateway + Public IP   (stable outbound IP for NSG whitelisting)
//   - Log Analytics Workspace   (receives all container stdout/stderr)
//   - Container Apps Environment (the runtime boundary for Container Apps)
//
// The NAT Gateway ensures all outbound traffic from the Container Apps
// Environment uses a single, stable public IP. This IP is whitelisted
// on the Neo4j NSG to allow Bolt (7687) connectivity.
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

@description('Name of the Virtual Network')
param vnetName string = 'mmct-vnet'

@description('Name of the Container Apps subnet')
param subnetName string = 'mmct-subnet-apps'

@description('VNet address space')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Subnet address range (minimum /23 for Container Apps)')
param subnetAddressPrefix string = '10.0.0.0/23'

@description('Name of the NAT Gateway')
param natGatewayName string = 'mmct-natgw'

@description('Name of the NAT Gateway public IP')
param natGatewayPipName string = 'mmct-natgw-pip'

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
// Networking — VNet + NAT Gateway
// ============================================================================

resource natGatewayPip 'Microsoft.Network/publicIPAddresses@2023-09-01' = {
  name: natGatewayPipName
  location: location
  sku: {
    name: 'Standard'
  }
  zones: ['1', '2', '3']
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

resource natGateway 'Microsoft.Network/natGateways@2023-09-01' = {
  name: natGatewayName
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    idleTimeoutInMinutes: 4
    publicIpAddresses: [
      { id: natGatewayPip.id }
    ]
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2023-09-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [vnetAddressPrefix]
    }
    subnets: [
      {
        name: subnetName
        properties: {
          addressPrefix: subnetAddressPrefix
          natGateway: {
            id: natGateway.id
          }
          delegations: [
            {
              name: 'Microsoft.App.environments'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
        }
      }
    ]
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
    vnetConfiguration: {
      infrastructureSubnetId: vnet.properties.subnets[0].id
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
output natGatewayPublicIp string = natGatewayPip.properties.ipAddress
output subnetId        string = vnet.properties.subnets[0].id