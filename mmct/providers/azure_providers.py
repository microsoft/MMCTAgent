from typing import Dict, Any, Optional, List
from azure.identity import DefaultAzureCredential, AzureCliCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
from openai import AsyncAzureOpenAI, AzureOpenAI
from loguru import logger

from .base import LLMProvider, EmbeddingProvider, SearchProvider, TranscriptionProvider, VisionProvider
from ..exceptions import ProviderException, ConfigurationException
from ..utils.error_handler import handle_exceptions, convert_exceptions


class AzureLLMProvider(LLMProvider):
    """Azure OpenAI LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = self._get_credential()
        self.client = self._initialize_client()
    
    def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        try:
            # Try Azure CLI credential first
            cli_credential = AzureCliCredential()
            # Test if CLI credential works by getting a token
            cli_credential.get_token("https://cognitiveservices.azure.com/.default")
            logger.info("Using Azure CLI credential")
            return cli_credential
        except Exception as e:
            logger.info(f"Azure CLI credential not available: {e}. Using DefaultAzureCredential")
            return DefaultAzureCredential()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            endpoint = self.config.get("endpoint")
            api_version = self.config.get("api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("use_managed_identity", True)
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            if not endpoint:
                raise ConfigurationException("Azure OpenAI endpoint is required")
            
            if use_managed_identity:
                token_provider = get_bearer_token_provider(
                    self.credential, 
                    "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    max_retries=max_retries,
                    timeout=timeout
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure OpenAI API key is required when managed identity is disabled")
                
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    max_retries=max_retries,
                    timeout=timeout
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Azure OpenAI."""
        try:
            deployment_name = self.config.get("deployment_name")
            if not deployment_name:
                raise ConfigurationException("Azure OpenAI deployment name is required")
            
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.0))
            max_tokens = kwargs.get("max_tokens", 4000)
            response_format = kwargs.get("response_format")
            
            # Remove temperature, max_tokens, and response_format from kwargs to avoid duplicate arguments
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "response_format"]}
            
            # Check if response_format is a BaseModel - if so, use parse() instead of create()
            from pydantic import BaseModel
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                response = await self.client.chat.completions.parse(
                    model=deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **filtered_kwargs
                )
                
                return {
                    "content": response.choices[0].message.parsed,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
            else:
                # Standard completion without structured output
                completion_kwargs = {
                    "model": deployment_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **filtered_kwargs
                }
                
                if response_format:
                    completion_kwargs["response_format"] = response_format
                
                response = await self.client.chat.completions.create(**completion_kwargs)
                
                return {
                    "content": response.choices[0].message.content,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
        except Exception as e:
            logger.error(f"Azure OpenAI chat completion failed: {e}")
            raise ProviderException(f"Azure OpenAI chat completion failed: {e}")


class AzureEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = self._get_credential()
        self.client = self._initialize_client()
    
    def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        try:
            # Try Azure CLI credential first
            cli_credential = AzureCliCredential()
            # Test if CLI credential works by getting a token
            cli_credential.get_token("https://cognitiveservices.azure.com/.default")
            logger.info("Using Azure CLI credential")
            return cli_credential
        except Exception as e:
            logger.info(f"Azure CLI credential not available: {e}. Using DefaultAzureCredential")
            return DefaultAzureCredential()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            endpoint = self.config.get("endpoint")
            api_version = self.config.get("api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("use_managed_identity", True)
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            if not endpoint:
                raise ConfigurationException("Azure OpenAI endpoint is required")
            
            if use_managed_identity:
                token_provider = get_bearer_token_provider(
                    self.credential, 
                    "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    max_retries=max_retries,
                    timeout=timeout
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure OpenAI API key is required when managed identity is disabled")
                
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    max_retries=max_retries,
                    timeout=timeout
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using Azure OpenAI."""
        try:
            deployment_name = self.config.get("deployment_name") or self.config.get("embedding_deployment_name")
            if not deployment_name:
                raise ConfigurationException(
                    "Azure OpenAI embedding deployment name is required. "
                    "Set EMBEDDING_SERVICE_DEPLOYMENT_NAME environment variable."
                )
            
            response = await self.client.embeddings.create(
                model=deployment_name,
                input=text,
                **kwargs
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure OpenAI embedding failed: {e}")
            raise ProviderException(f"Azure OpenAI embedding failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts using Azure OpenAI."""
        try:
            deployment_name = self.config.get("deployment_name") or self.config.get("embedding_deployment_name")
            if not deployment_name:
                raise ConfigurationException(
                    "Azure OpenAI embedding deployment name is required. "
                    "Set EMBEDDING_SERVICE_DEPLOYMENT_NAME environment variable."
                )
            
            response = await self.client.embeddings.create(
                model=deployment_name,
                input=texts,
                **kwargs
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Azure OpenAI batch embedding failed: {e}")
            raise ProviderException(f"Azure OpenAI batch embedding failed: {e}")


class AzureSearchProvider(SearchProvider):
    """Azure AI Search provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        try:
            self.credential = AzureCliCredential()
            self.credential.get_token("https://search.azure.com/.default")
        except Exception as e:
            logger.info(f"Azure CLI credential not available: {e}. Using DefaultAzureCredential")
            # Fallback to DefaultAzureCredential if CLI credential is not available
            self.credential = DefaultAzureCredential()
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure AI Search client."""
        try:
            endpoint = self.config.get("endpoint")
            index_name = self.config.get("index_name", "default")
            use_managed_identity = self.config.get("use_managed_identity", True)
            
            if not endpoint:
                raise ConfigurationException("Azure AI Search endpoint is required")
            
            if use_managed_identity:
                return SearchClient(
                    endpoint=endpoint,
                    index_name=index_name,
                    credential=self.credential
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure AI Search API key is required when managed identity is disabled")
                
                from azure.core.credentials import AzureKeyCredential
                return SearchClient(
                    endpoint=endpoint,
                    index_name=index_name,
                    credential=AzureKeyCredential(api_key)
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure AI Search client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search(self, query: str, index_name: str = None, **kwargs) -> List[Dict]:
        """Search documents using Azure AI Search."""
        try:
            vector_queries = None
            semantic_configuration_name=None

            search_text = kwargs.pop("search_text", query)
            top = kwargs.pop("top", 10)
            embedding = kwargs.pop("embedding", [])
            query_type = kwargs.pop("query_type", None)
            vector_queries = kwargs.pop("vector_queries",None)

            if query_type=="semantic":
                semantic_configuration_name=kwargs.pop("semantic_configuration_name","my-semantic-search-config")
                search_text = None
                
            if query_type=="vector":
                query_type = None
                
            if embedding and top and not vector_queries:
                vector_query = VectorizedQuery(
                    vector=embedding, k_nearest_neighbors=top, fields="embeddings"
                )
                vector_queries = [vector_query]

            if index_name and index_name != self.client._index_name:
                # Create new client for different index
                config = self.config.copy()
                config["index_name"] = index_name
                client = AzureSearchProvider(config).client
            else:
                client = self.client
            
            results = client.search(
                search_text=search_text,
                top=top,
                query_type=query_type,
                vector_queries=vector_queries,
                semantic_configuration_name=semantic_configuration_name,
                **kwargs
            )
            
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Azure AI Search failed: {e}")
            raise ProviderException(f"Azure AI Search failed: {e}")
        
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_document(self, document: Dict, index_name: str = None) -> bool:
        """Index a document in Azure AI Search."""
        try:
            if index_name and index_name != self.client._index_name:
                # Create new client for different index
                config = self.config.copy()
                config["index_name"] = index_name
                client = AzureSearchProvider(config).client
            else:
                client = self.client
            
            result = client.upload_documents(documents=[document])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Azure AI Search indexing failed: {e}")
            raise ProviderException(f"Azure AI Search indexing failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_document(self, doc_id: str, index_name: str = None) -> bool:
        """Delete a document from Azure AI Search."""
        try:
            if index_name and index_name != self.client._index_name:
                # Create new client for different index
                config = self.config.copy()
                config["index_name"] = index_name
                client = AzureSearchProvider(config).client
            else:
                client = self.client
            
            result = client.delete_documents(documents=[{"id": doc_id}])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Azure AI Search deletion failed: {e}")
            raise ProviderException(f"Azure AI Search deletion failed: {e}")


class AzureTranscriptionProvider(TranscriptionProvider):
    """Azure Speech Service transcription provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        try:
            self.credential = AzureCliCredential()
            self.credential.get_token("https://cognitiveservices.azure.com/.default")
        except Exception as e:
            logger.info(f"Azure CLI credential not available: {e}. Using DefaultAzureCredential")
            # Fallback to DefaultAzureCredential if CLI credential is not available
            self.credential = DefaultAzureCredential()
        self.speech_config = self._initialize_speech_config()
    
    def _initialize_speech_config(self):
        """Initialize Azure Speech configuration."""
        try:
            endpoint = self.config.get("endpoint")
            use_managed_identity = self.config.get("use_managed_identity", True)
            
            if not endpoint:
                raise ConfigurationException("Azure Speech Service endpoint is required")
            
            if use_managed_identity:
                # For managed identity, we need to use token authentication
                # This is a simplified implementation
                return SpeechConfig(
                    endpoint=endpoint,
                    auth_token=self._get_auth_token()
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure Speech Service API key is required when managed identity is disabled")
                
                return SpeechConfig(
                    endpoint=endpoint,
                    subscription=api_key
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure Speech config: {e}")
    
    def _get_auth_token(self):
        """Get authentication token for Speech Service."""
        # This is a simplified implementation
        # In practice, you'd need to implement proper token management
        return "dummy_token"
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """Transcribe audio bytes using Azure Speech Service."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to handle audio data properly
            raise NotImplementedError("Direct audio data transcription not implemented")
        except Exception as e:
            logger.error(f"Azure Speech Service transcription failed: {e}")
            raise ProviderException(f"Azure Speech Service transcription failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio file using Azure Speech Service."""
        try:
            if language:
                self.speech_config.speech_recognition_language = language
            
            audio_config = AudioConfig(filename=audio_path)
            recognizer = SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            result = recognizer.recognize_once()
            
            if result.reason == result.reason.RecognizedSpeech:
                return result.text
            else:
                raise ProviderException(f"Speech recognition failed: {result.reason}")
        except Exception as e:
            logger.error(f"Azure Speech Service file transcription failed: {e}")
            raise ProviderException(f"Azure Speech Service file transcription failed: {e}")


class AzureVisionProvider(VisionProvider):
    """Azure Computer Vision provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # This would be implemented with Azure Computer Vision SDK
        # For now, we'll use Azure OpenAI Vision capabilities
        self.llm_provider = AzureLLMProvider(config)
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def analyze_image(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """Analyze image using Azure Computer Vision."""
        try:
            # This is a simplified implementation using Azure OpenAI Vision
            # In practice, you'd use Azure Computer Vision SDK
            raise NotImplementedError("Azure Computer Vision analysis not implemented")
        except Exception as e:
            logger.error(f"Azure Computer Vision analysis failed: {e}")
            raise ProviderException(f"Azure Computer Vision analysis failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def extract_text(self, image_data: bytes, **kwargs) -> str:
        """Extract text from image using Azure Computer Vision."""
        try:
            # This is a simplified implementation
            # In practice, you'd use Azure Computer Vision OCR
            raise NotImplementedError("Azure Computer Vision OCR not implemented")
        except Exception as e:
            logger.error(f"Azure Computer Vision text extraction failed: {e}")
            raise ProviderException(f"Azure Computer Vision text extraction failed: {e}")