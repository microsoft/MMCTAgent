# Neo4j Production VM - Bicep Deployment

This folder contains **Azure Developer CLI (azd)** compatible Bicep templates for deploying **Neo4j Community Edition 5.26** on an Azure VM.

## Tested Configuration

| Component | Version |
|-----------|---------|
| **Neo4j** | 5.26.0 |
| **APOC** | 5.26.0 |
| **GDS** | 2.13.2 |

## Quick Start

```bash
cd config/neo4j_prod

# Deploy
azd up

# You'll be prompted for:
# - Environment name (e.g., "neo4j-prod")
# - Azure subscription
# - Azure region
# - adminPasswordOrKey (VM SSH password)
# - neo4jPassword (Neo4j database password)
```

## Files

| File | Description |
|------|-------------|
| `azure.yaml` | azd project configuration |
| `infra/main.bicep` | Bicep template for Neo4j VM infrastructure |
| `infra/main.parameters.json` | Default parameter values |

## Architecture

```
                    ┌─────────────────────────────────┐
                    │   Public Load Balancer (opt)    │
                    │     bolt://<PUBLIC_IP>:7687     │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┴─────────────────┐
                    │   Internal Load Balancer        │
                    │      bolt://10.0.1.10:7687      │
                    └───────────────┬─────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │   neo4j-core-1    │
                          │  Neo4j 5.26.0     │
                          │  D16s_v5 VM       │
                          │  APOC + GDS       │
                          ├───────────────────┤
                          │  Premium SSD Disk │
                          │  /var/lib/neo4j   │
                          │   (Persistent)    │
                          └───────────────────┘
```

## Resources Created

- **1x Virtual Machine** (Standard_D16s_v5 - 16 vCPUs, 64GB RAM)
- **1x Premium SSD Data Disk** (1TB default) - Persistent storage
- **1x Virtual Network** with subnet and NAT Gateway
- **1x Internal Load Balancer** (Standard SKU)
- **1x Public Load Balancer** (optional, when `enablePublicIP=true`)
- **1x Network Security Group**

## Plugins

| Plugin | Version | Purpose |
|--------|---------|---------|
| **APOC** | 5.26.0 | Advanced procedures (data import/export, utilities) |
| **GDS** | 2.13.2 | Graph algorithms, ML, KNN vector search |

## Vector Search Support

1. **Native HNSW Indexes (Neo4j 5.x)**:
   ```cypher
   CREATE VECTOR INDEX chapter_embedding IF NOT EXISTS
   FOR (c:Chapter)
   ON (c.embedding)
   OPTIONS {
     indexConfig: {
       `vector.dimensions`: 384,
       `vector.similarity_function`: 'cosine'
     }
   };
   ```

2. **GDS KNN Procedures**:
   ```cypher
   CALL gds.knn.stream('myGraph', {
     nodeProperties: ['embedding'],
     topK: 10
   })
   ```

## Memory Configuration

Default settings for D16s_v5 (64GB RAM):

| Setting | Value | Purpose |
|---------|-------|---------|
| `heap.initial_size` | 31G | JVM heap |
| `heap.max_size` | 31G | JVM heap max |
| `pagecache.size` | 24G | Graph data cache |

Supports **200-400 concurrent queries**.

## Connection

After deployment, get connection URIs:

```bash
azd env get-values | grep neo4j
```

**With public IP enabled:**
```
http://<PUBLIC_IP>:7474  # Neo4j Browser
bolt://<PUBLIC_IP>:7687  # Bolt connection
```

**Private access (within VNet):**
```
http://10.0.1.10:7474
bolt://10.0.1.10:7687
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vmSize` | Standard_D16s_v5 | VM size |
| `dataDiskSizeGB` | 1024 | Data disk size |
| `heapSize` | 31G | Neo4j heap size |
| `pagecacheSize` | 24G | Neo4j page cache |
| `enablePublicIP` | true | Enable public access |
| `enableApocPlugin` | true | Install APOC |
| `enableGdsPlugin` | true | Install GDS |

## Estimated Cost (Southeast Asia)

| Resource | SKU | Cost |
|----------|-----|------|
| 1x VM | D16s_v5 | ~$560 |
| 1x Premium SSD | P30 (1TB) | ~$135 |
| Load Balancer | Standard | ~$20 |
| NAT Gateway | Standard | ~$45 |
| **Total** | | **~$760/mo** |

## Troubleshooting

### Check Neo4j status
```bash
az vm run-command invoke \
  --resource-group <RG> \
  --name neo4j-core-1 \
  --command-id RunShellScript \
  --scripts "systemctl status neo4j"
```

### View logs
```bash
az vm run-command invoke \
  --resource-group <RG> \
  --name neo4j-core-1 \
  --command-id RunShellScript \
  --scripts "cat /var/lib/neo4j/logs/neo4j.log | tail -50"
```

### Verify plugins
```bash
az vm run-command invoke \
  --resource-group <RG> \
  --name neo4j-core-1 \
  --command-id RunShellScript \
  --scripts "/opt/neo4j/bin/cypher-shell -u neo4j -p '<PASSWORD>' 'RETURN gds.version(), apoc.version()'"
```

## Clean Up

```bash
azd down
```
