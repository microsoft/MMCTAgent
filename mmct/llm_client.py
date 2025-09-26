import os
import warnings
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
        # Deprecation warning
        warnings.warn(
            "LLMClient is deprecated and will be removed in a future version. "
            "Use the provider pattern instead: "
            "from mmct.providers.factory import provider_factory; "
            "provider = provider_factory.create_llm_provider(provider_name, config)",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Use Azure CLI credential if available, fallback to DefaultAzureCredential
        self.credential = self._get_credential()
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
    
    def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        try:
            from azure.identity import AzureCliCredential
            # Try Azure CLI credential first
            cli_credential = AzureCliCredential()
            # Test if CLI credential works by getting a token
            cli_credential.get_token("https://cognitiveservices.azure.com/.default")
            return cli_credential
        except Exception:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
            
    def  _initialize_openai_stt_client(self):
        if self.service_provider == "azure":
            managed_identity_setting = os.environ.get("LLM_USE_MANAGED_IDENTITY", os.environ.get("MANAGED_IDENTITY", None))
            if managed_identity_setting is None:
                raise Exception(
                    "LLM_USE_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
                
            # Use LLM_ENDPOINT for Azure OpenAI Whisper, fallback to SPEECH_SERVICE_ENDPOINT for Azure Speech Service
            azure_endpoint = os.environ.get("LLM_ENDPOINT", None) or os.environ.get("SPEECH_SERVICE_ENDPOINT", None)
            api_version = os.environ.get("LLM_API_VERSION", None) or os.environ.get("SPEECH_SERVICE_API_VERSION", None)
            deployment_name = os.environ.get("SPEECH_SERVICE_DEPLOYMENT_NAME", None)
            if None in [azure_endpoint, deployment_name, api_version]:
                raise Exception("Required Azure OpenAI credentials are missing for azure openai stt client!")
            
            self.azure_managed_identity = managed_identity_setting.upper()
            
            if self.azure_managed_identity == "TRUE":
                azure_ad_token_provider=get_bearer_token_provider(self.credential, "https://cognitiveservices.azure.com/.default")
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
                # Use LLM_API_KEY for Azure OpenAI Whisper, fallback to SPEECH_SERVICE_KEY for Azure Speech Service
                api_key = os.environ.get("LLM_API_KEY", None) or os.environ.get("SPEECH_SERVICE_KEY", None)
                if api_key is None:
                    raise Exception(
                        "Required Azure OpenAI API Key for initializing OpenAI STT client!"
                    )
                if self.isAsync:
                    return AsyncAzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_deployment = deployment_name,
                                api_key=api_key)
                else:
                    return AzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_version=api_version,
                                azure_deployment = deployment_name,
                                api_key=api_key)
                    
        else:
            api_key = os.environ.get("OPENAI_SPEECH_SERVICE_KEY")
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
            managed_identity_setting = os.environ.get("EMBEDDING_USE_MANAGED_IDENTITY", os.environ.get("MANAGED_IDENTITY", None))
            if managed_identity_setting is None:
                raise Exception(
                    "EMBEDDING_USE_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
                
            azure_endpoint=os.environ.get("EMBEDDING_SERVICE_ENDPOINT", None)
            api_version=os.environ.get("EMBEDDING_SERVICE_API_VERSION",None)
            deployment_name = os.environ.get("EMBEDDING_SERVICE_DEPLOYMENT_NAME",None)
            if None in [azure_endpoint, deployment_name, api_version]:
                raise Exception("Required Azure OpenAI credentials are missing for embedding client!")
            self.azure_managed_identity = managed_identity_setting.upper()
            if self.azure_managed_identity == "TRUE":
                azure_ad_token_provider=get_bearer_token_provider(self.credential, "https://cognitiveservices.azure.com/.default")
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
                AZURE_OPENAI_EMBEDDING_KEY = os.environ.get("EMBEDDING_SERVICE_API_KEY", None)
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
            managed_identity_setting = os.environ.get("LLM_USE_MANAGED_IDENTITY", os.environ.get("MANAGED_IDENTITY", None))
            if managed_identity_setting is None:
                raise Exception(
                    "LLM_USE_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
            return self._initialize_azure_client()
        else:
            return self._initialize_openai_client()
        

    def _initialize_azure_client(self):
        """Initializes an Azure OpenAI client."""
        azure_endpoint = os.environ.get("LLM_ENDPOINT", None)
        deployment_name = os.environ.get("LLM_VISION_DEPLOYMENT_NAME", None)
        api_version = os.environ.get("LLM_VISION_API_VERSION", None)
        if None in [azure_endpoint, deployment_name, api_version]:
            raise Exception("Required Azure OpenAI credentials are missing")
        managed_identity_setting = os.environ.get("LLM_USE_MANAGED_IDENTITY", os.environ.get("MANAGED_IDENTITY", ""))
        self.azure_managed_identity = managed_identity_setting.upper()
        if self.azure_managed_identity == "TRUE":
            token_provider = get_bearer_token_provider(
                self.credential, "https://cognitiveservices.azure.com/.default"
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
            AZURE_OPENAI_GPT4V_KEY = os.environ.get("LLM_API_KEY", None)
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
            managed_identity_setting = os.environ.get("LLM_USE_MANAGED_IDENTITY", os.environ.get("MANAGED_IDENTITY", None))
            if managed_identity_setting is None:
                raise Exception(
                    "LLM_USE_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )
            azure_endpoint = os.getenv("LLM_ENDPOINT", None)
            deployment_name = os.getenv("LLM_DEPLOYMENT_NAME", None)
            model = os.getenv("LLM_MODEL_NAME",None)
            api_version = os.getenv("LLM_API_VERSION", None)
            if None in [azure_endpoint, deployment_name, api_version, model]:
                raise Exception("Required Azure OpenAI credentials are missing")
            self.azure_managed_identity = managed_identity_setting.upper()
            if self.azure_managed_identity == "TRUE":

                token_provider = get_bearer_token_provider(
                    self.credential,
                    "https://cognitiveservices.azure.com/.default",
                )
                return AzureOpenAIChatCompletionClient(
                    azure_deployment=deployment_name,
                    model=deployment_name,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=200,
                    temperature=0
                )
            else:
                AZURE_OPENAI_GPT4V_KEY = os.environ.get("LLM_API_KEY", None)
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