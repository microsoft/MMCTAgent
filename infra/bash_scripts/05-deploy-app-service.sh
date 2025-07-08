# fetching the variables from the source file
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the env vars using absolute path
source "$script_dir/00-setup-env-vars.sh"

export MSYS_NO_PATHCONV=1

# fetching the resource id for premium app service plan
echo "Fetching resource ID for $aspPremiumName..."
serverfarmsASPId=$(az resource show \
    --resource-group "$resourceGroup" \
    --name "$aspPremiumName" \
    --resource-type "Microsoft.Web/serverfarms" \
    --query "id" \
    --output tsv)

echo "✅ Found resource ID: $serverfarmsASPId"

clientIdOfIdentity=$(az identity show \
  --resource-group $resourceGroup \
  --name $identityName \
  --query clientId \
  --output tsv)

# client of the the managed identity resource
echo "Client Id: $clientIdOfIdentity"

subscriptionId=$(az account show --query id -o tsv)
# subscription id
echo "SubscriptionId: $subscriptionId"

# creating the user assigned identity scope
userAssignedIdentityScope="subscriptions/$subscriptionId/resourceGroups/$resourceGroup/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$identityName"

echo "userAssignedIdentityScope: $userAssignedIdentityScope"

# fetching the id for the speech resource
speechResourceId=$(az resource show \
    --resource-group "$resourceGroup" \
    --name "$azureSpeechServiceName" \
    --resource-type "Microsoft.CognitiveServices/accounts" \
    --query "id" \
    --output tsv)

# Deploying the app services
echo "Checking if main app service $mainAppServiceName exists...."

if az resource show \
    --name "$mainAppServiceName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.Web/sites" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $mainAppServiceName"
else
    echo "⚙️ Resource does not exist, creating it..."
    # Optional: check if file exists
    if [ ! -f "$mainAppTemplateFile" ]; then
        echo "File not found: $mainAppTemplateFile"
        exit 1
    fi

    az deployment group create \
      --resource-group $resourceGroup \
      --template-file "$mainAppTemplateFile" \
      --parameters mainAppName=$mainAppServiceName \
      serverfarmsASPId=$serverfarmsASPId \
      mainAppImageandTag=$mainAppImageandTag \
      clientIdOfIdentity=$clientIdOfIdentity \
      userAssignedIdentityScope=$userAssignedIdentityScope \
      storageAccountName=$storageAccountName \
      speechResourceId=$speechResourceId \
      speechRegion=$azureSpeechServiceRegion \
      azureOpenAIName=$azureOpenAIName \
      aiSearchServiceName=$aiSearchServiceName \
      eventhubName=$eventhubName \
      --debug
fi