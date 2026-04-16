// Neo4j 5.26.0 Enterprise — Single VM Deployment on Azure
//
// Use this template for:  Enterprise edition only (hardcoded)
//                         For Community, use main.bicep instead
//
// Enterprise extras over main.bicep:
//   - server.cluster.system_database_mode=PRIMARY + initial.server.mode_constraint=PRIMARY
//     (explicit single-instance mode — prevents cluster bootstrap ambiguity)
//   - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes always set in systemd unit
//   - Optional paid license file via neo4jLicenseKeyBase64 parameter
//   - Optional APOC Extended plugin (Enterprise-only procedures)
//   - Optional Prometheus metrics endpoint on port 2004
//   - NSG rule for port 2004 (VNet-scoped metrics scraping)
//
// Recommended deploy path (reads neo4j_config.yaml automatically):
//   python infra/neo4j/deploy.py
//
// Direct deploy:
//   az deployment group create \
//     --resource-group <rg> \
//     --name neo4j-enterprise \
//     --template-file infra/neo4j/bicep/main-enterprise.bicep \
//     --parameters infra/neo4j/bicep/main-enterprise.parameters.json \
//     --parameters neo4jPassword=<pwd> adminPasswordOrKey=<pwd> location=<region>
//
// ── Parameters fixed in this template (cannot change without editing bicep) ─
//   Edition        Always 'enterprise' — no parameter
//   Neo4j version  Always 5.26.0
//   OS             Always Ubuntu 22.04 LTS Gen2
//   Java           Always OpenJDK 17
//   APOC version   Always 5.26.0
//   GDS version    Always 2.13.2
//
// ── Parameters users commonly change ─────────────────────────────────────────
//   vmSize                VM SKU (default: Standard_D16s_v5 — 16 vCPU / 64 GB RAM)
//   dataDiskSizeGB        Data disk size in GB (256 / 512 / 1024 / 2048; default: 1024)
//   heapSize              JVM heap (default: 31G — tune to ~50% of VM RAM)
//   pagecacheSize         Neo4j page cache (default: 24G — tune to fit working set)
//   enablePublicIP        Expose via public load balancer (default: false)
//   enableApocPlugin      Install APOC Core (default: true)
//   enableApocExtended    Install APOC Extended — Enterprise-only (default: false)
//   enableGdsPlugin       Install Graph Data Science (default: true)
//   enableMetrics         Prometheus metrics on port 2004 (default: true)
//   neo4jLicenseKeyBase64 Base64-encoded paid license file; blank = eval license
// ─────────────────────────────────────────────────────────────────────────────

targetScope = 'resourceGroup'

// ============================================================================
// Parameters
// ============================================================================

@description('Azure region for deployment')
param location string = 'southeastasia'

@description('Prefix for Neo4j VM names')
param vmNamePrefix string = 'neo4j-ent'

@description('Admin username for the VM')
param adminUsername string = 'azureuser'

@secure()
@description('Admin password or SSH public key for the VM')
param adminPasswordOrKey string

@secure()
@description('Password for Neo4j database authentication')
param neo4jPassword string

@secure()
@description('''
Optional — base64-encoded content of a Neo4j Enterprise license file.
Leave blank to use the evaluation/developer license
(NEO4J_ACCEPT_LICENSE_AGREEMENT=yes).

To encode your license file:
  base64 -w 0 neo4j.license

Provide the resulting string here. The script will write it to
/opt/neo4j/licenses/neo4j.license inside the VM.
''')
param neo4jLicenseKeyBase64 string = ''

@description('VM size for Neo4j node (D16s_v5 recommended for high concurrency)')
@allowed([
  'Standard_D4s_v5'
  'Standard_D8s_v5'
  'Standard_D16s_v5'
  'Standard_D32s_v5'
  'Standard_E16s_v5'
  'Standard_E32s_v5'
])
param vmSize string = 'Standard_D16s_v5'

@description('Size of the data disk for Neo4j storage')
@allowed([256, 512, 1024, 2048])
param dataDiskSizeGB int = 1024

@description('Neo4j heap size (initial and max)')
param heapSize string = '31G'

@description('Neo4j page cache size')
param pagecacheSize string = '24G'

@description('Virtual network address prefix')
param vnetAddressPrefix string = '10.0.0.0/16'

@description('Subnet address prefix for Neo4j')
param subnetAddressPrefix string = '10.0.1.0/24'

@description('Private IP address for the internal load balancer')
param loadBalancerPrivateIP string = '10.0.1.10'

@description('Enable public IP for external access (NOT recommended for production)')
param enablePublicIP bool = false

@description('Enable APOC Core plugin for advanced procedures')
param enableApocPlugin bool = true

@description('Enable APOC Extended plugin (Enterprise-only — extra procedures beyond APOC Core)')
param enableApocExtended bool = false

@description('Enable Graph Data Science plugin for ML and vector search')
param enableGdsPlugin bool = true

@description('Enable Prometheus metrics endpoint on port 2004 (Enterprise feature)')
param enableMetrics bool = true

// ============================================================================
// Variables
// ============================================================================

var vnetName              = '${vmNamePrefix}-vnet'
var subnetName            = '${vmNamePrefix}-subnet'
var lbName                = '${vmNamePrefix}-lb'
var nsgName               = '${vmNamePrefix}-nsg'
var lbFrontendName        = 'neo4j-frontend'
var lbPublicFrontendName  = 'neo4j-public-frontend'
var lbBackendPoolName     = 'neo4j-backend'
var lbProbeBoltName       = 'bolt-probe'
var lbProbeHttpName       = 'http-probe'
var publicLbName          = '${vmNamePrefix}-lb-public'

// Plugin versions (tested compatible with Neo4j 5.26)
var apocVersion = '5.26.0'
var gdsVersion  = '2.13.2'

// ============================================================================
// Network Security Group
// ============================================================================

resource nsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: nsgName
  location: location
  properties: {
    securityRules: concat([
      {
        name: 'Allow-Bolt-VNet'
        properties: {
          priority: 1100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '7687'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationAddressPrefix: '*'
          description: 'Allow Bolt protocol from VNet'
        }
      }
      {
        name: 'Allow-HTTP-VNet'
        properties: {
          priority: 1200
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '7474'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationAddressPrefix: '*'
          description: 'Allow HTTP for Neo4j Browser from VNet'
        }
      }
      {
        name: 'Allow-Metrics-VNet'
        properties: {
          priority: 1300
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '2004'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationAddressPrefix: '*'
          description: 'Allow Prometheus metrics scrape from VNet (Enterprise)'
        }
      }
    ], enablePublicIP ? [
      {
        name: 'Allow-Bolt-Public'
        properties: {
          priority: 1101
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '7687'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: '*'
          description: 'Allow Bolt protocol from Internet (public access)'
        }
      }
      {
        name: 'Allow-HTTP-Public'
        properties: {
          priority: 1201
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '7474'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: '*'
          description: 'Allow HTTP for Neo4j Browser from Internet (public access)'
        }
      }
    ] : [])
  }
}

// ============================================================================
// NAT Gateway for outbound internet connectivity
// ============================================================================

resource natGatewayPublicIP 'Microsoft.Network/publicIPAddresses@2023-04-01' = {
  name: '${vmNamePrefix}-natgw-pip'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
    publicIPAddressVersion: 'IPv4'
  }
}

resource natGateway 'Microsoft.Network/natGateways@2023-04-01' = {
  name: '${vmNamePrefix}-natgw'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    idleTimeoutInMinutes: 10
    publicIpAddresses: [
      {
        id: natGatewayPublicIP.id
      }
    ]
  }
}

// ============================================================================
// Virtual Network
// ============================================================================

resource vnet 'Microsoft.Network/virtualNetworks@2023-04-01' = {
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
          networkSecurityGroup: {
            id: nsg.id
          }
          natGateway: {
            id: natGateway.id
          }
        }
      }
    ]
  }
}

// ============================================================================
// Load Balancer Public IP (optional)
// ============================================================================

resource lbPublicIP 'Microsoft.Network/publicIPAddresses@2023-04-01' = if (enablePublicIP) {
  name: '${vmNamePrefix}-lb-pip'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
    publicIPAddressVersion: 'IPv4'
  }
}

// ============================================================================
// Internal Load Balancer (private access)
// ============================================================================

resource lb 'Microsoft.Network/loadBalancers@2023-04-01' = {
  name: lbName
  location: location
  sku: {
    name: 'Standard'
    tier: 'Regional'
  }
  properties: {
    frontendIPConfigurations: [
      {
        name: lbFrontendName
        properties: {
          privateIPAddress: loadBalancerPrivateIP
          privateIPAllocationMethod: 'Static'
          subnet: {
            id: vnet.properties.subnets[0].id
          }
        }
      }
    ]
    backendAddressPools: [
      {
        name: lbBackendPoolName
      }
    ]
    loadBalancingRules: [
      {
        name: 'bolt-rule'
        properties: {
          frontendIPConfiguration: {
            id: resourceId('Microsoft.Network/loadBalancers/frontendIPConfigurations', lbName, lbFrontendName)
          }
          backendAddressPool: {
            id: resourceId('Microsoft.Network/loadBalancers/backendAddressPools', lbName, lbBackendPoolName)
          }
          protocol: 'Tcp'
          frontendPort: 7687
          backendPort: 7687
          enableFloatingIP: false
          idleTimeoutInMinutes: 15
          probe: {
            id: resourceId('Microsoft.Network/loadBalancers/probes', lbName, lbProbeBoltName)
          }
          loadDistribution: 'SourceIP'
        }
      }
      {
        name: 'http-rule'
        properties: {
          frontendIPConfiguration: {
            id: resourceId('Microsoft.Network/loadBalancers/frontendIPConfigurations', lbName, lbFrontendName)
          }
          backendAddressPool: {
            id: resourceId('Microsoft.Network/loadBalancers/backendAddressPools', lbName, lbBackendPoolName)
          }
          protocol: 'Tcp'
          frontendPort: 7474
          backendPort: 7474
          enableFloatingIP: false
          idleTimeoutInMinutes: 15
          probe: {
            id: resourceId('Microsoft.Network/loadBalancers/probes', lbName, lbProbeHttpName)
          }
        }
      }
    ]
    probes: [
      {
        name: lbProbeBoltName
        properties: {
          protocol: 'Tcp'
          port: 7687
          intervalInSeconds: 15
          numberOfProbes: 2
        }
      }
      {
        name: lbProbeHttpName
        properties: {
          protocol: 'Tcp'
          port: 7474
          intervalInSeconds: 15
          numberOfProbes: 2
        }
      }
    ]
  }
}

// ============================================================================
// Public Load Balancer (optional — external access)
// ============================================================================

resource lbPublic 'Microsoft.Network/loadBalancers@2023-04-01' = if (enablePublicIP) {
  name: publicLbName
  location: location
  sku: {
    name: 'Standard'
    tier: 'Regional'
  }
  properties: {
    frontendIPConfigurations: [
      {
        name: lbPublicFrontendName
        properties: {
          publicIPAddress: {
            id: lbPublicIP.id
          }
        }
      }
    ]
    backendAddressPools: [
      {
        name: lbBackendPoolName
      }
    ]
    loadBalancingRules: [
      {
        name: 'bolt-rule-public'
        properties: {
          frontendIPConfiguration: {
            id: resourceId('Microsoft.Network/loadBalancers/frontendIPConfigurations', publicLbName, lbPublicFrontendName)
          }
          backendAddressPool: {
            id: resourceId('Microsoft.Network/loadBalancers/backendAddressPools', publicLbName, lbBackendPoolName)
          }
          protocol: 'Tcp'
          frontendPort: 7687
          backendPort: 7687
          enableFloatingIP: false
          idleTimeoutInMinutes: 15
          probe: {
            id: resourceId('Microsoft.Network/loadBalancers/probes', publicLbName, lbProbeBoltName)
          }
          loadDistribution: 'SourceIP'
        }
      }
      {
        name: 'http-rule-public'
        properties: {
          frontendIPConfiguration: {
            id: resourceId('Microsoft.Network/loadBalancers/frontendIPConfigurations', publicLbName, lbPublicFrontendName)
          }
          backendAddressPool: {
            id: resourceId('Microsoft.Network/loadBalancers/backendAddressPools', publicLbName, lbBackendPoolName)
          }
          protocol: 'Tcp'
          frontendPort: 7474
          backendPort: 7474
          enableFloatingIP: false
          idleTimeoutInMinutes: 15
          probe: {
            id: resourceId('Microsoft.Network/loadBalancers/probes', publicLbName, lbProbeHttpName)
          }
        }
      }
    ]
    probes: [
      {
        name: lbProbeBoltName
        properties: {
          protocol: 'Tcp'
          port: 7687
          intervalInSeconds: 15
          numberOfProbes: 2
        }
      }
      {
        name: lbProbeHttpName
        properties: {
          protocol: 'Tcp'
          port: 7474
          intervalInSeconds: 15
          numberOfProbes: 2
        }
      }
    ]
  }
}

// ============================================================================
// Network Interface
// ============================================================================

var privateBackendPool = {
  id: resourceId('Microsoft.Network/loadBalancers/backendAddressPools', lbName, lbBackendPoolName)
}

var publicBackendPool = {
  id: resourceId('Microsoft.Network/loadBalancers/backendAddressPools', publicLbName, lbBackendPoolName)
}

resource nic 'Microsoft.Network/networkInterfaces@2023-04-01' = {
  name: '${vmNamePrefix}-nic-1'
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          privateIPAllocationMethod: 'Dynamic'
          subnet: {
            id: vnet.properties.subnets[0].id
          }
          loadBalancerBackendAddressPools: enablePublicIP ? [privateBackendPool, publicBackendPool] : [privateBackendPool]
        }
      }
    ]
    enableAcceleratedNetworking: true
  }
  dependsOn: enablePublicIP ? [lb, lbPublic] : [lb]
}

// ============================================================================
// Virtual Machine
// ============================================================================

resource vm 'Microsoft.Compute/virtualMachines@2023-03-01' = {
  name: '${vmNamePrefix}-1'
  location: location
  properties: {
    hardwareProfile: {
      vmSize: vmSize
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-jammy'
        sku: '22_04-lts-gen2'
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
        diskSizeGB: 128
      }
      dataDisks: [
        {
          lun: 0
          name: '${vmNamePrefix}-1-data'
          createOption: 'Empty'
          diskSizeGB: dataDiskSizeGB
          managedDisk: {
            storageAccountType: 'Premium_LRS'
          }
          caching: 'ReadOnly'
        }
      ]
    }
    osProfile: {
      computerName: '${vmNamePrefix}-1'
      adminUsername: adminUsername
      adminPassword: adminPasswordOrKey
      linuxConfiguration: {
        disablePasswordAuthentication: false
        provisionVMAgent: true
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic.id
        }
      ]
    }
    diagnosticsProfile: {
      bootDiagnostics: {
        enabled: true
      }
    }
  }
}

// ============================================================================
// VM Extension — Neo4j Enterprise Setup Script
// ============================================================================

var enableApocString         = enableApocPlugin    ? 'true' : 'false'
var enableApocExtendedString = enableApocExtended  ? 'true' : 'false'
var enableGdsString          = enableGdsPlugin     ? 'true' : 'false'
var enableMetricsString      = enableMetrics       ? 'true' : 'false'
var hasLicenseKey            = !empty(neo4jLicenseKeyBase64)
var hasLicenseKeyString      = hasLicenseKey       ? 'true' : 'false'
var publicIpAddress          = lbPublicIP.?properties.ipAddress ?? loadBalancerPrivateIP

var setupScript = '''#!/bin/bash
set -e

# ── Variables injected from Bicep ──────────────────────────────────────────
NEO4J_PASSWORD="__NEO4J_PASSWORD__"
LICENSE_KEY_BASE64="__LICENSE_KEY_BASE64__"
HAS_LICENSE_KEY="__HAS_LICENSE_KEY__"
HEAP_SIZE="__HEAP_SIZE__"
PAGECACHE_SIZE="__PAGECACHE_SIZE__"
ENABLE_APOC="__ENABLE_APOC__"
ENABLE_APOC_EXTENDED="__ENABLE_APOC_EXTENDED__"
ENABLE_GDS="__ENABLE_GDS__"
ENABLE_METRICS="__ENABLE_METRICS__"
APOC_VERSION="__APOC_VERSION__"
GDS_VERSION="__GDS_VERSION__"
ADVERTISED_ADDRESS="__ADVERTISED_ADDRESS__"

echo "=== Starting Neo4j Enterprise 5.26.0 Installation ==="

# ── Find and mount the data disk ────────────────────────────────────────────
echo "Finding data disk..."
DATA_DISK=""

for disk in /dev/sd[a-z]; do
  if [ -b "$disk" ]; then
    PART_COUNT=$(lsblk -n "$disk" | wc -l)
    if [ "$PART_COUNT" -eq 1 ]; then
      DATA_DISK="$disk"
      echo "Found unpartitioned disk: $DATA_DISK"
      break
    fi
  fi
done

if [ -z "$DATA_DISK" ]; then
  echo "ERROR: Could not find data disk"
  lsblk
  exit 1
fi

echo "Using data disk: $DATA_DISK"

if mountpoint -q /var/lib/neo4j; then
  echo "Data disk already mounted at /var/lib/neo4j"
else
  if ! blkid "$DATA_DISK" > /dev/null 2>&1; then
    echo "Formatting data disk..."
    sudo mkfs.ext4 -F "$DATA_DISK"
  fi

  sudo mkdir -p /var/lib/neo4j
  sudo mount "$DATA_DISK" /var/lib/neo4j

  if ! grep -q "$DATA_DISK" /etc/fstab; then
    echo "$DATA_DISK /var/lib/neo4j ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
  fi
fi

# ── Install Java 17 ─────────────────────────────────────────────────────────
echo "Installing Java 17..."
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk unzip curl

# ── Download Neo4j 5.26.0 Enterprise ────────────────────────────────────────
echo "Downloading Neo4j Enterprise 5.26.0..."
cd /opt
sudo curl -L -o neo4j-enterprise-5.26.0-unix.tar.gz \
  "https://dist.neo4j.org/neo4j-enterprise-5.26.0-unix.tar.gz"
sudo tar -xzf neo4j-enterprise-5.26.0-unix.tar.gz
sudo rm  neo4j-enterprise-5.26.0-unix.tar.gz
sudo rm -f /opt/neo4j
sudo ln -s /opt/neo4j-enterprise-5.26.0 /opt/neo4j

# ── Create neo4j user ────────────────────────────────────────────────────────
id neo4j 2>/dev/null || sudo useradd -r -s /bin/false neo4j

# ── License file ─────────────────────────────────────────────────────────────
echo "Configuring license..."
sudo mkdir -p /opt/neo4j/licenses

if [ "$HAS_LICENSE_KEY" = "true" ] && [ -n "$LICENSE_KEY_BASE64" ]; then
  echo "Writing paid license file to /opt/neo4j/licenses/neo4j.license..."
  echo "$LICENSE_KEY_BASE64" | base64 -d | sudo tee /opt/neo4j/licenses/neo4j.license > /dev/null
  echo "Paid license file written."
else
  echo "No paid license provided — evaluation license will be accepted via env var."
fi

# ── Download plugins ─────────────────────────────────────────────────────────
echo "Downloading plugins..."
sudo mkdir -p /opt/neo4j/plugins

if [ "$ENABLE_APOC" = "true" ]; then
  echo "Installing APOC Core ${APOC_VERSION}..."
  sudo curl -L -o /opt/neo4j/plugins/apoc-${APOC_VERSION}-core.jar \
    "https://github.com/neo4j/apoc/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-core.jar"
fi

if [ "$ENABLE_APOC_EXTENDED" = "true" ]; then
  echo "Installing APOC Extended ${APOC_VERSION} (Enterprise-only)..."
  sudo curl -L -o /opt/neo4j/plugins/apoc-${APOC_VERSION}-extended.jar \
    "https://github.com/neo4j/apoc/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-extended.jar"
fi

if [ "$ENABLE_GDS" = "true" ]; then
  echo "Installing GDS ${GDS_VERSION}..."
  sudo curl -L -o /opt/neo4j/plugins/neo4j-graph-data-science-${GDS_VERSION}.jar \
    "https://github.com/neo4j/graph-data-science/releases/download/${GDS_VERSION}/neo4j-graph-data-science-${GDS_VERSION}.jar"
fi

# ── Configure Neo4j Enterprise ───────────────────────────────────────────────
echo "Configuring Neo4j Enterprise..."
METRICS_BLOCK=""
if [ "$ENABLE_METRICS" = "true" ]; then
  METRICS_BLOCK="
# Enterprise Metrics (Prometheus) — using Neo4j 5.x key names
server.metrics.enabled=true
server.metrics.prometheus.enabled=true
server.metrics.prometheus.endpoint=localhost:2004"
fi

sudo tee /opt/neo4j/conf/neo4j.conf > /dev/null <<EOF
# ── Edition ────────────────────────────────────────────────────────────────
# Explicit single-instance mode — Neo4j 5.26 equivalents for deprecated dbms.mode=SINGLE
server.cluster.system_database_mode=PRIMARY
initial.server.mode_constraint=PRIMARY

# ── Memory ────────────────────────────────────────────────────────────────
server.memory.heap.initial_size=$HEAP_SIZE
server.memory.heap.max_size=$HEAP_SIZE
server.memory.pagecache.size=$PAGECACHE_SIZE

# ── Data directories — use persistent disk ────────────────────────────────
server.directories.data=/var/lib/neo4j/data
server.directories.logs=/var/lib/neo4j/logs

# ── Connectors ────────────────────────────────────────────────────────────
server.bolt.enabled=true
server.bolt.listen_address=0.0.0.0:7687
server.bolt.advertised_address=${ADVERTISED_ADDRESS}:7687
server.http.enabled=true
server.http.listen_address=0.0.0.0:7474
server.http.advertised_address=${ADVERTISED_ADDRESS}:7474
server.https.enabled=false

# ── Security ──────────────────────────────────────────────────────────────
dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=apoc.*,gds.*
dbms.security.procedures.allowlist=apoc.*,gds.*

# ── Bolt thread pool ──────────────────────────────────────────────────────
server.bolt.thread_pool_min_size=5
server.bolt.thread_pool_max_size=400

# ── Query settings ────────────────────────────────────────────────────────
db.transaction.timeout=30s
db.lock.acquisition.timeout=10s

# ── Logging ───────────────────────────────────────────────────────────────
db.logs.query.enabled=INFO
db.logs.query.threshold=1s
$METRICS_BLOCK
EOF

# ── Create data directories and set permissions ───────────────────────────────
sudo mkdir -p /var/lib/neo4j/data /var/lib/neo4j/logs /var/lib/neo4j/metrics
sudo mkdir -p /opt/neo4j/run /opt/neo4j/metrics
sudo chown -R neo4j:neo4j /var/lib/neo4j
sudo chown -R neo4j:neo4j /opt/neo4j

# ── Set initial password ─────────────────────────────────────────────────────
sudo /opt/neo4j/bin/neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD"

# ── Create systemd service ───────────────────────────────────────────────────
sudo tee /etc/systemd/system/neo4j.service > /dev/null <<EOF
[Unit]
Description=Neo4j Enterprise Graph Database
After=network.target

[Service]
Type=simple
User=neo4j
Group=neo4j
Environment="NEO4J_HOME=/opt/neo4j"
Environment="NEO4J_ACCEPT_LICENSE_AGREEMENT=yes"
ExecStart=/opt/neo4j/bin/neo4j console
Restart=on-failure
LimitNOFILE=60000
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

# ── Start Neo4j ──────────────────────────────────────────────────────────────
echo "Starting Neo4j Enterprise..."
sudo systemctl daemon-reload
sudo systemctl enable neo4j
sudo systemctl start neo4j

echo "Waiting for Neo4j to start..."
sleep 30

echo "=== Neo4j Enterprise 5.26.0 Installation Complete ==="
echo "    APOC Core=${ENABLE_APOC}  APOC Extended=${ENABLE_APOC_EXTENDED}  GDS=${ENABLE_GDS}"
echo "    Metrics=${ENABLE_METRICS}"
echo "    License=${HAS_LICENSE_KEY} (true=paid file, false=eval env var)"
echo "    Advertised address: ${ADVERTISED_ADDRESS}"
'''

var scriptWithParams = replace(
  replace(
    replace(
      replace(
        replace(
          replace(
            replace(
              replace(
                replace(
                  replace(
                    replace(setupScript,
                      '__NEO4J_PASSWORD__',       neo4jPassword),
                      '__LICENSE_KEY_BASE64__',   neo4jLicenseKeyBase64),
                      '__HAS_LICENSE_KEY__',      hasLicenseKeyString),
                      '__HEAP_SIZE__',            heapSize),
                      '__PAGECACHE_SIZE__',       pagecacheSize),
                      '__ENABLE_APOC__',          enableApocString),
                      '__ENABLE_APOC_EXTENDED__', enableApocExtendedString),
                      '__ENABLE_GDS__',           enableGdsString),
                      '__ENABLE_METRICS__',       enableMetricsString),
                      '__APOC_VERSION__',         apocVersion),
                      '__GDS_VERSION__',          gdsVersion)

// Final replacement outside the chain to keep line length reasonable
var scriptFinal = replace(scriptWithParams, '__ADVERTISED_ADDRESS__', publicIpAddress)

resource vmExtension 'Microsoft.Compute/virtualMachines/extensions@2023-03-01' = {
  parent: vm
  name: 'neo4j-enterprise-setup'
  location: location
  properties: {
    publisher: 'Microsoft.Azure.Extensions'
    type: 'CustomScript'
    typeHandlerVersion: '2.1'
    autoUpgradeMinorVersion: true
    settings: {
      skipDos2Unix: false
    }
    protectedSettings: {
      script: base64(scriptFinal)
    }
  }
}

// ============================================================================
// Outputs
// ============================================================================

output neo4jBoltUri          string = 'bolt://${loadBalancerPrivateIP}:7687'
output neo4jHttpUri          string = 'http://${loadBalancerPrivateIP}:7474'
output neo4jPublicBoltUri    string = enablePublicIP ? 'bolt://${lbPublicIP.?properties.ipAddress}:7687'  : 'N/A - Public IP disabled'
output neo4jPublicHttpUri    string = enablePublicIP ? 'http://${lbPublicIP.?properties.ipAddress}:7474'  : 'N/A - Public IP disabled'
output metricsEndpoint       string = enableMetrics  ? 'http://${loadBalancerPrivateIP}:2004/metrics'    : 'N/A - Metrics disabled'
output vnetId                string = vnet.id
output subnetId              string = vnet.properties.subnets[0].id
output loadBalancerPrivateIP string = loadBalancerPrivateIP
output loadBalancerPublicIP  string = lbPublicIP.?properties.ipAddress ?? 'N/A'
output vmName                string = vm.name
output licenseMode           string = hasLicenseKey ? 'paid-license-file' : 'evaluation-env-var'
