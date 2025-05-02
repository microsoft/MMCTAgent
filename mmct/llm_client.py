import os
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from azure.identity import DefaultAzureCredential
from openai import OpenAI, AzureOpenAI, AsyncAzureOpenAI, AsyncOpenAI
from azure.identity import get_bearer_token_provider
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(),override=True)


class LLMClient:
    def __init__(self, autogen=False, service_provider="azure", embedding=False, stt=False, isAsync = False):
        self.service_provider = service_provider
        self.isAsync = isAsync
        if autogen:
            self.client = self._initialize_autogen_client()
        elif embedding:
            self.client = self._initialize_embedding_client()
        elif stt:
            self.client = self._initialize_openai_stt_client()
        else:
            self.client = self._initialize_client()
            
    def  _initialize_openai_stt_client(self):
        if self.service_provider == "azure":
            if os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY", None) is None:
                raise Exception(
                    "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
                
            azure_endpoint=os.environ.get("AZURE_OPENAI_STT_ENDPOINT", None)
            api_version=os.environ.get("AZURE_OPENAI_STT_API_VERSION",None)
            deployment_name = os.environ.get("AZURE_OPENAI_STT_DEPLOYMENT",None)
            if None in [azure_endpoint, deployment_name, api_version]:
                raise Exception("Required Azure OpenAI credentials are missing for azure openai stt client!")
            
            self.azure_managed_identity = os.environ.get(
                "AZURE_OPENAI_MANAGED_IDENTITY", ""
            ).upper()
            
            if self.azure_managed_identity == "TRUE":
                azure_ad_token_provider=get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
                if self.isAsync:
                    return AsyncAzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_ad_token_provider=azure_ad_token_provider,
                                timeout=200)
                else:
                    return AzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_ad_token_provider=azure_ad_token_provider)
                    
            else:
                AZURE_OPENAI_STT_KEY = os.environ.get("AZURE_OPENAI_STT_KEY", None)
                if AZURE_OPENAI_STT_KEY is None:
                    raise Exception(
                        "Required Azure OpenAI API Key for initializing OpenAI STT client!"
                    )
                if self.isAsync:
                    return AsyncAzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_deployment = deployment_name,
                                api_key=AZURE_OPENAI_STT_KEY)
                else:
                    return AzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_deployment = deployment_name,
                                api_key=AZURE_OPENAI_STT_KEY)
                    
        else:
            api_key = os.environ.get("OPENAI_STT_KEY")
            if api_key is None:
                    raise Exception(
                        "Required OpenAI API Key for initializing OpenAI STT client!"
                    )
            if self.isAsync:
                return AsyncOpenAI(
                    api_key=api_key
                )
            else:
                return OpenAI(
                    api_key=api_key
                )
            

    def _initialize_embedding_client(self):
        if self.service_provider == "azure":
            if os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY", None) is None:
                raise Exception(
                    "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
                
            azure_endpoint=os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT", None)
            api_version=os.environ.get("AZURE_EMBEDDING_API_VERSION",None)
            deployment_name = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT",None)
            if None in [azure_endpoint, deployment_name, api_version]:
                raise Exception("Required Azure OpenAI credentials are missing for embedding client!")
            self.azure_managed_identity = os.environ.get(
                "AZURE_OPENAI_MANAGED_IDENTITY", ""
            ).upper()
            if self.azure_managed_identity == "TRUE":
                azure_ad_token_provider=get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
                if self.isAsync:
                    return AsyncAzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_ad_token_provider=azure_ad_token_provider,
                                azure_deployment = deployment_name,
                                timeout=200)
                else:
                    return AzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_ad_token_provider=azure_ad_token_provider,
                                azure_deployment = deployment_name,
                                timeout=200)
            else:
                AZURE_OPENAI_EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY", None)
                if AZURE_OPENAI_EMBEDDING_KEY is None:
                    raise Exception(
                        "Required Azure OpenAI API Key for initializing OpenAI Embedding client!"
                    )
                if self.isAsync:
                    return AsyncAzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_deployment = deployment_name,
                                api_key=AZURE_OPENAI_EMBEDDING_KEY,
                                timeout=200)
                else:
                    return AzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_deployment = deployment_name,
                                api_key=AZURE_OPENAI_EMBEDDING_KEY,
                                timeout=200)
                    
        else:
            api_key = os.environ.get("OPENAI_EMBEDDING_KEY")
            if api_key is None:
                    raise Exception(
                        "Required OpenAI API Key for initializing OpenAI Embedding client!"
                    )
            if self.isAsync:
                return AsyncOpenAI(
                    api_key=api_key,
                    timeout=200
                )
            else:
                return OpenAI(
                    api_key=api_key,
                    timeout=200
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
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)
        deployment_name = os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT", None)
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
            if self.isAsync:
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    max_retries=2,
                    timeout=200
                )
            else:
                return AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    max_retries=2,
                    timeout=200
                )
        else:
            AZURE_OPENAI_GPT4V_KEY = os.environ.get("AZURE_OPENAI_KEY", None)
            if AZURE_OPENAI_GPT4V_KEY is None:
                raise Exception(
                    "Required Azure OpenAI API Key for initializing OpenAI client!"
                )
            if self.isAsync:
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    api_key=AZURE_OPENAI_GPT4V_KEY,
                    max_retries=2,
                    timeout=200
                )
            else:
                return AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    api_key=AZURE_OPENAI_GPT4V_KEY,
                    max_retries=2,
                    timeout=200
                )

    def _initialize_openai_client(self):
        """Initializes an OpenAI client."""
        openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_api_key is None:
            raise Exception(
                "Required Azure OpenAI API Key for initializing OpenAI client!"
            )
        if self.isAsync:
            return AsyncOpenAI(api_key=openai_api_key, timeout=200)
        else:
            return OpenAI(api_key=openai_api_key, timeout=200)

    def _initialize_autogen_client(self):
        """Initializes an AutoGen-compatible client."""
        if self.service_provider == "azure":
            if os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY", None) is None:
                raise Exception(
                    "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", None)
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", None)
            model = os.getenv("AZURE_OPENAI_MODEL",None)
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", None)
            if None in [azure_endpoint, deployment_name, api_version, model]:
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
                    model=model,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=200,
                    temperature=0
                )
            else:
                AZURE_OPENAI_GPT4V_KEY = os.environ.get("AZURE_OPENAI_KEY", None)
                return AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment_name,
                    model=deployment_name,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    api_key=AZURE_OPENAI_GPT4V_KEY,
                    timeout=200,
                    temperature=0
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