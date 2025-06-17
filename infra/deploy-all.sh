#!/bin/bash
set -e  # exit immediately on error

bashScriptsDirName='bash_scripts'

# Deploy the resources
./$bashScriptsDirName/01-deploy-infra.sh
# ./$bashScriptsDirName/02-create-managed-identity.sh
# ./$bashScriptsDirName/04-deploy-app-service.sh
# ./$bashScriptsDirName/05-deploy-container-app.sh