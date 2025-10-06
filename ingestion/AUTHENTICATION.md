# Authentication Guide

This guide explains the authentication mechanisms used in the Lively ingestion pipeline.

## Authentication Priority Order

The pipeline uses a **prioritized authentication chain** for maximum flexibility and security:

### 🥇 Priority 1: Azure CLI Credentials (Recommended)
- **Best for**: Local development, testing
- **Setup**: `az login`
- **Pros**: No secrets in code, easy to use, secure
- **Cons**: Requires Azure CLI installed

### 🥈 Priority 2: DefaultAzureCredential
- **Best for**: Production, Azure-hosted services
- **Includes**: Managed Identity, Environment Variables, Visual Studio Code, Azure PowerShell
- **Pros**: Works automatically in Azure, no code changes needed
- **Cons**: Requires proper Azure setup

### 🥉 Priority 3: API Keys / Connection Strings (Fallback)
- **Best for**: Legacy systems, quick testing
- **Pros**: Simple, no Azure setup needed
- **Cons**: Less secure, requires secret management

---

## Quick Start

### Option 1: Azure CLI (Recommended)

```bash
# 1. Install Azure CLI (if not already installed)
# Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# 2. Login to Azure
az login

# 3. Set your subscription (if you have multiple)
az account set --subscription "Your Subscription Name"

# 4. Run your code - authentication happens automatically!
python your_script.py
```

**Environment variables needed:**
```bash
# .env file
SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
AZURE_STORAGE_ACCOUNT_URL=https://youraccount.blob.core.windows.net
```

**Python code:**
```python
import asyncio
from ingestion import run_ingestion

async def main():
    # No API keys needed - uses Azure CLI automatically!
    result = await run_ingestion(
        video_path="video.mp4",
        search_endpoint="https://your-search.search.windows.net",
        storage_account_url="https://youraccount.blob.core.windows.net",
        index_name="keyframes"
    )

asyncio.run(main())
```

---

### Option 2: Managed Identity (Production)

**For Azure VMs, App Services, Container Instances, etc.**

```python
import asyncio
from ingestion import run_ingestion

async def main():
    # Managed Identity is detected automatically!
    result = await run_ingestion(
        video_path="video.mp4",
        search_endpoint="https://your-search.search.windows.net",
        storage_account_url="https://youraccount.blob.core.windows.net",
        index_name="keyframes"
    )

asyncio.run(main())
```

**Azure setup:**
```bash
# Enable Managed Identity on your resource
az vm identity assign --name myVM --resource-group myRG

# Grant permissions
az role assignment create \
  --assignee <managed-identity-principal-id> \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.Storage/storageAccounts/<account>

az role assignment create \
  --assignee <managed-identity-principal-id> \
  --role "Search Index Data Contributor" \
  --scope /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.Search/searchServices/<service>
```

---

### Option 3: API Keys / Connection Strings (Fallback)

```bash
# .env file
SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
SEARCH_API_KEY=your_api_key

AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...
```

```python
import asyncio
import os
from ingestion import run_ingestion

async def main():
    result = await run_ingestion(
        video_path="video.mp4",
        search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
        search_api_key=os.getenv("SEARCH_API_KEY"),  # Fallback
        storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),  # Fallback
        index_name="keyframes"
    )

asyncio.run(main())
```

---

## How It Works

### Authentication Chain

```python
# Internal authentication flow (simplified)

def get_credential(api_key=None):
    if api_key:
        return AzureKeyCredential(api_key)

    try:
        # Try Azure CLI first
        cli_cred = AzureCliCredential()
        cli_cred.get_token("...")  # Test it
        return cli_cred
    except:
        pass

    # Fall back to DefaultAzureCredential
    return DefaultAzureCredential()
```

### For Blob Storage

```python
# Priority order for blob storage:

if connection_string:
    # Use connection string
    client = BlobServiceClient.from_connection_string(conn_str)
elif storage_account_url:
    # Use chained credential (CLI -> Default)
    credential = ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential()
    )
    client = BlobServiceClient(storage_account_url, credential)
else:
    raise ValueError("Need either connection_string or storage_account_url")
```

### For Azure AI Search

```python
# Priority order for search:

if api_key:
    credential = AzureKeyCredential(api_key)
else:
    # Chained credential (CLI -> Default)
    credential = ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential()
    )

client = SearchClient(endpoint, index_name, credential)
```

---

## Troubleshooting

### Azure CLI Authentication Fails

**Error**: `Azure CLI authentication not available`

**Solutions**:
1. Install Azure CLI: https://aka.ms/install-azure-cli
2. Login: `az login`
3. Verify: `az account show`
4. Set subscription: `az account set --subscription "..."`

### Managed Identity Fails

**Error**: `ManagedIdentityCredential authentication failed`

**Solutions**:
1. Verify Managed Identity is enabled:
   ```bash
   az vm identity show --name myVM --resource-group myRG
   ```

2. Check role assignments:
   ```bash
   az role assignment list --assignee <principal-id>
   ```

3. Grant necessary roles:
   - `Storage Blob Data Contributor` for Blob Storage
   - `Search Index Data Contributor` for AI Search

### No Credentials Found

**Error**: `No valid Azure credentials found`

**Solutions**:
1. Quick fix: Use API keys (connection string)
2. Recommended: Run `az login`
3. Production: Set up Managed Identity

---

## Best Practices

### 🔒 Security

1. **Never commit secrets** to version control
2. **Use Azure CLI** for development
3. **Use Managed Identity** for production
4. **Rotate keys regularly** if using API keys
5. **Use Azure Key Vault** for secret management

### 🚀 Performance

1. **Azure CLI is fast** - credential caching
2. **Managed Identity is faster** - no external calls
3. **Connection strings bypass auth** - fastest but least secure

### 📝 Environment Configuration

**Development (.env.local)**
```bash
# Use Azure CLI - no secrets needed!
SEARCH_SERVICE_ENDPOINT=https://dev-search.search.windows.net
AZURE_STORAGE_ACCOUNT_URL=https://devaccount.blob.core.windows.net
```

**Production (Environment Variables)**
```bash
# Managed Identity - set these in Azure Portal
SEARCH_SERVICE_ENDPOINT=https://prod-search.search.windows.net
AZURE_STORAGE_ACCOUNT_URL=https://prodaccount.blob.core.windows.net
```

**Testing (.env.test)**
```bash
# Can use keys for quick tests
SEARCH_SERVICE_ENDPOINT=https://test-search.search.windows.net
SEARCH_API_KEY=test_key
AZURE_STORAGE_CONNECTION_STRING=...test_connection_string...
```

---

## Testing Authentication

### Test Azure CLI

```python
from ingestion import auth

# Test if Azure CLI works
credential = auth.get_azure_credential()
is_valid = auth.test_credential(credential)
print(f"Azure CLI authentication: {'✓' if is_valid else '✗'}")
```

### Test Blob Storage

```python
from ingestion import BlobStorageManager

async def test_blob():
    try:
        manager = BlobStorageManager(
            storage_account_url="https://youraccount.blob.core.windows.net"
        )
        print("✓ Blob Storage authentication successful")
    except Exception as e:
        print(f"✗ Blob Storage authentication failed: {e}")
```

### Test Azure AI Search

```python
from ingestion import KeyframeSearchIndex

async def test_search():
    try:
        index = KeyframeSearchIndex(
            search_endpoint="https://your-search.search.windows.net",
            index_name="test"
        )
        print("✓ AI Search authentication successful")
    except Exception as e:
        print(f"✗ AI Search authentication failed: {e}")
```

---

## Migration Guide

### From Connection String to Azure CLI

**Before:**
```python
result = await run_ingestion(
    video_path="video.mp4",
    search_endpoint="...",
    storage_connection_string="DefaultEndpointsProtocol=https;...",
    search_api_key="your_key",
    index_name="keyframes"
)
```

**After:**
```bash
# One-time setup
az login
```

```python
# Simpler, more secure code
result = await run_ingestion(
    video_path="video.mp4",
    search_endpoint="...",
    storage_account_url="https://account.blob.core.windows.net",
    index_name="keyframes"
)
```

---

## FAQ

**Q: Do I need to install anything extra?**
A: For Azure CLI auth, install Azure CLI. For Managed Identity, nothing extra needed.

**Q: Can I mix authentication methods?**
A: Yes! You can use CLI for storage and API key for search, or any combination.

**Q: Will this work in Azure DevOps pipelines?**
A: Yes! Use service connections and Managed Identity in Azure DevOps.

**Q: What about local development?**
A: Azure CLI is perfect for local development - just run `az login`.

**Q: Is this slower than connection strings?**
A: Slightly, but the security benefits far outweigh the minimal performance difference.

**Q: Can I disable CLI auth and force API keys?**
A: Yes, just provide `api_key` or `connection_string` parameters explicitly.

---

## Additional Resources

- [Azure CLI Documentation](https://docs.microsoft.com/en-us/cli/azure/)
- [Managed Identity Overview](https://docs.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/overview)
- [DefaultAzureCredential](https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential)
- [Azure RBAC Roles](https://docs.microsoft.com/en-us/azure/role-based-access-control/built-in-roles)
