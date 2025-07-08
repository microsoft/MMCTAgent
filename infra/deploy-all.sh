#!/bin/bash
set -e

CONFIG_FILE="./infra_config.yaml"
bashScriptsDirName="bash_scripts"

get_yaml_value() {
    python -c "import yaml, sys; print(yaml.safe_load(open('$CONFIG_FILE'))$1)"
}

# 1. deploy infrastructure
if [[ "$(get_yaml_value "['deployInfra']['enabled']")" == "True" ]]; then
    echo "==> Running Infra Deployment..."
    bash ./$bashScriptsDirName/01-deploy-infra.sh
else
    echo "=> Skipping infrastructure deployment...!"
fi

# 2. create env
if [[ "$(get_yaml_value "['envCreation']['enabled']")" == "True" ]]; then
    echo "==> Creating ENV..."
    bash ./$bashScriptsDirName/02-generate-env.sh
else
    echo "=> Skipping ENV Creation...!"
fi

# 3. Creating managed identity resource
if [[ "$(get_yaml_value "['midentityCreation']['enabled']")" == "True" ]]; then
    echo "==> Creating Managed Identity Resource..."
    bash ./$bashScriptsDirName/03-create-managed-identity.sh
else
    echo "=> Skipping managed identity resource creation...!"
fi

# 4. building and pushing images
if [[ "$(get_yaml_value "['buildAndPushImagesToACR']['enabled']")" == "True" ]]; then
    echo "==> Building and pushing images..."
    bash ./$bashScriptsDirName/04-build-push-docker-images.sh
else
    echo "=> Skipping building and pushing of docker images...!"
fi

# 5. deploy app services
if [[ "$(get_yaml_value "['deployAppService']['enabled']")" == "True" ]]; then
    echo "==> Deploying App Services..."
    bash ./$bashScriptsDirName/05-deploy-app-service.sh
else
    echo "=> Skipping the deployment of app services...!"
fi

# 6. deploy container apps
if [[ "$(get_yaml_value "['deployContainerApps']['enabled']")" == "True" ]]; then
    echo "==> Deploy container apps..."
    bash ./$bashScriptsDirName/06-deploy-container-app.sh
else
    echo "=> Skipping the deployment of container apps...!"
fi