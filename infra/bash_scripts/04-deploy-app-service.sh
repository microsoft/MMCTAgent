# fetching the variables from the source file
source 00-setup-env-vars.sh
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

# validating the certain variables whether available or not
echo "Image and tag: $ingestionProducerImageAndTag"

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
echo "Checking if ingestion app service $ingestionProducerAppName exists...."

if az resource show \
    --name "$ingestionProducerAppName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.Web/sites" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $ingestionProducerAppName"
else
    echo "⚙️ Resource does not exist, creating it..."
    az deployment group create \
      --resource-group $resourceGroup \
      --template-file $ingestionProducerTemplateFile \
      --parameters ingestionProducerAppName=$ingestionProducerAppName \
      serverfarmsASPId=$serverfarmsASPId \
      ingestionProducerImageAndTag=$ingestionProducerImageAndTag \
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


echo "Checking if query app service $queryProducerAppName exists...."

if az resource show \
    --name "$queryProducerAppName" \
    --resource-group "$resourceGroup" \
    --resource-type "Microsoft.Web/sites" \
    --query "id" \
    --output tsv >/dev/null 2>&1; then
    echo "✅ Resource exists: Name = $queryProducerAppName"
else
    echo "⚙️ Resource does not exist, creating it..."
    az deployment group create \
      --resource-group $resourceGroup \
      --template-file $queryProducerTemplateFile \
      --parameters queryProducerAppName=$queryProducerAppName \
      serverfarmsASPId=$serverfarmsASPId \
      queryProducerImageAndTag=$queryProducerImageAndTag \
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

