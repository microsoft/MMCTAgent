# ARM Template Infrastructure Deployment Guide

This repository contains Bash scripts and ARM templates to automate the deployment of Azure infrastructure components like App Services, Container Apps, Managed Identity, and Container Registries.

---

## 📋 Prerequisites

Before using this setup, ensure the following tools are installed and configured:

### 🔧 Tools Required

- **Azure CLI**: [Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Bash Shell**:  
  - **Windows**: Use [Git Bash](https://git-scm.com/downloads) or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
  - **Linux/macOS**: Native Bash supported

### 🔐 Azure Authentication

```bash
az login
az account set --subscription "<your-subscription-name-or-id>"
```

### 🔓 Make Scripts Executable (Linux/macOS/WSL)

```bash
chmod +x 00-setup-env-vars.sh
chmod +x 01-deploy-infra.sh
chmod +x 02-create-managed-identity.sh
chmod +x 04-deploy-app-service.sh
chmod +x 05-deploy-container-app.sh
```

> On Windows, use Git Bash or Windows Subsystem for Linux (WSL) to execute scripts.
If you're using PowerShell, adapt the syntax accordingly or use WSL for better compatibility.

## 🧭 Flow of Execution

1. Set up the environment variables
2. Deploy the necessary/required resources.
3. Create the Managed Identity Resource.
4. Build the docker images and push to Container Registry
5. Deploy App services.
6. Deploy the Container Apps.

### **1. Setup the env/variable name**

- Add relevant names of the required resources and container images in the `00-setup-env-vars.sh`
- Ensure the names do not exceed the character limit and naming convention form for each resource.
- `Tip: use lowercase characters under 24 characters without any special characters`

#### **2. Deploy the Resources**

- Deploy the resources (except App Services, Container Apps and Identity Resource) by running `01-deploy-infra.sh` file.

> `01-deploy-infra.sh` file fetches the names of resources from `00-setup-env-vars.sh`, check the existance of the respective resource and if the resource do not exists, then creates it.

```bash
./01-deploy-infra.sh
```
