#!/bin/bash

# Set the environment variables
export BLOB_CONTAINER_NAME=""
#Set this as "True" or "False" based on whether you want to use managed identity with blob
#If you set this as "True", then you need to fill BLOB1 else fill BLOB2
export BLOB_MANAGED_IDENTITY = ""
#BLOB1
export BLOB_ACCOUNT_URL = ""
#BLOB2
export BLOB_CONNECTION_STRING=""
export BLOB_SAS_TOKEN=""


export AZURECV_ENDPOINT=""
#Set this as "True" or "False" based on whether you want to use managed identity with Azure CV
export AZURECV_MANAGED_IDENTITY = ""
#If "False" then fill the below
export AZURECV_KEY=""


export AZURE_OPENAI_GPT4_ENDPOINT=""
export GPT4_32K_DEPLOYMENT=""
export AZURE_OPENAI_GPT4V_ENDPOINT=""
export GPT4V_DEPLOYMENT=""
export AZURE_OPENAI_EMBEDDING_ENDPOINT=""
export ADA_EMBEDDING_DEPLOYMENT=""
export AZURE_OPENAI_WHISPER_ENDPOINT=""
export WHISPER_DEPLOYMENT=""
#Set this as "True" or "False" based on whether you want to use managed identity with Azure OpenAI
export AZURE_OPENAI_MANAGED_IDENTITY = ""
#If "False" then fill the below
export AZURE_OPENAI_GPT4_KEY = ""
export AZURE_OPENAI_GPT4V_KEY = ""
export AZURE_OPENAI_EMBEDDING_KEY = ""
export AZURE_OPENAI_WHISPER_KEY = ""


export AZURE_MODERATION_ENDPOINT = ""
#Set this as "True" or "False" based on whether you want to use managed identity with Azure Moderation
export AZURE_MODERATION_MANAGED_IDENTITY = ""
#If "False" then fill the below
export AZURE_MODERATION_KEY = ""