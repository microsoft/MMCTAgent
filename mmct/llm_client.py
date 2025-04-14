import os
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from azure.identity import DefaultAzureCredential
from openai import OpenAI, AzureOpenAI
from azure.identity import get_bearer_token_provider
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(),override=True)


class LLMClient:
    def __init__(self, autogen=False, service_provider="azure", embedding=False):
        self.service_provider = service_provider
        if autogen:
            self.client = self._initialize_autogen_client()
        elif embedding:
            self.client = self._initialize_embedding_client()
        else:
            self.client = self._initialize_client()

    def _initialize_embedding_client(self):
        if self.service_provider == "azure":
            if os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY", None) is None:
                raise Exception(
                    "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
                
            
            azure_endpoint=os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT", None)
            api_version=os.environ.get("EMBEDDING_API_VERSION",None)
            deployment_name = os.environ.get("EMBEDDING_DEPLOYMENT",None)
            if None in [azure_endpoint, deployment_name, api_version]:
                raise Exception("Required Azure OpenAI credentials are missing for embedding client!")
            self.azure_managed_identity = os.environ.get(
                "AZURE_OPENAI_MANAGED_IDENTITY", ""
            ).upper()
            if self.azure_managed_identity == "TRUE":
                azure_ad_token_provider=get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
                return AzureOpenAI(
                            azure_endpoint=azure_endpoint,
                            api_version=api_version,
                            azure_ad_token_provider=azure_ad_token_provider)
            else:
                AZURE_OPENAI_EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY", None)
                if AZURE_OPENAI_EMBEDDING_KEY is None:
                    raise Exception(
                        "Required Azure OpenAI API Key for initializing OpenAI Embedding client!"
                    )
                return AzureOpenAI(
                            azure_endpoint=azure_endpoint,
                            api_version=api_version,
                            azure_deployment = deployment_name,
                            api_key=AZURE_OPENAI_EMBEDDING_KEY)
                
        else:
            api_key = os.environ.get("OPENAI_EMBEDDING_KEY")
            return OpenAI(
                api_key=api_key
            )
    
    def _initialize_client(self):
        """Initializes either an Azure OpenAI client or an OpenAI client based on environment variables."""
        if self.service_provider == "azure":
            if os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY", None) is None:
                raise Exception(
                    "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
            return self._initialize_azure_client()
        else:
            return self._initialize_openai_client()
        

    def _initialize_azure_client(self):
        """Initializes an Azure OpenAI client."""
        azure_endpoint = os.environ.get("AZURE_OPENAI_GPT4V_ENDPOINT", None)
        deployment_name = os.environ.get("GPT4V_DEPLOYMENT", None)
        api_version = os.environ.get("AZURE_OPENAI_VISION_API_VERSION", None)
        if None in [azure_endpoint, deployment_name, api_version]:
            raise Exception("Required Azure OpenAI credentials are missing")
        self.azure_managed_identity = os.environ.get(
                "AZURE_OPENAI_MANAGED_IDENTITY", ""
            ).upper()
        if self.azure_managed_identity == "TRUE":
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            return AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
                max_retries=2,
            )
        else:
            AZURE_OPENAI_GPT4V_KEY = os.environ.get("AZURE_OPENAI_GPT4V_KEY", None)
            if AZURE_OPENAI_GPT4V_KEY is None:
                raise Exception(
                    "Required Azure OpenAI API Key for initializing OpenAI client!"
                )
            return AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=AZURE_OPENAI_GPT4V_KEY,
                max_retries=2,
            )

    def _initialize_openai_client(self):
        """Initializes an OpenAI client."""
        openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_api_key is None:
            raise Exception(
                "Required Azure OpenAI API Key for initializing OpenAI client!"
            )

        return OpenAI(api_key=openai_api_key)

    def _initialize_autogen_client(self):
        """Initializes an AutoGen-compatible client."""
        if self.service_provider == "azure":
            if os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY", None) is None:
                raise Exception(
                    "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
            azure_endpoint = os.getenv("AZURE_OPENAI_GPT4V_ENDPOINT", None)
            deployment_name = os.getenv("GPT4V_DEPLOYMENT", None)
            api_version = os.getenv("AZURE_OPENAI_VISION_API_VERSION", None)
            if None in [azure_endpoint, deployment_name, api_version]:
                raise Exception("Required Azure OpenAI credentials are missing")
            self.azure_managed_identity = os.environ.get(
                "AZURE_OPENAI_MANAGED_IDENTITY", ""
            ).upper()
            if self.azure_managed_identity == "TRUE":

                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                return AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment_name,
                    model=deployment_name,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=200,
                    temperature=0,
                )
            else:
                AZURE_OPENAI_GPT4V_KEY = os.environ.get("AZURE_OPENAI_GPT4V_KEY", None)
                return AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment_name,
                    model=deployment_name,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    api_key=AZURE_OPENAI_GPT4V_KEY,
                    timeout=200,
                    temperature=0,
                )

        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            deployment_name = os.getenv("GPT4V_DEPLOYMENT", None)
            if not api_key:
                raise Exception(
                    "OPENAI_API_KEY environment variable is required for OpenAI AutoGen client."
                )
            return OpenAIChatCompletionClient(
                api_key=api_key, timeout=200, model=deployment_name, temperature=0
            )
            
    def get_client(self):
        return self.client