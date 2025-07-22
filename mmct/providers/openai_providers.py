from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI, OpenAI
from loguru import logger

from .base import LLMProvider, EmbeddingProvider, TranscriptionProvider, VisionProvider
from ..exceptions import ProviderException, ConfigurationException
from ..utils.error_handler import handle_exceptions, convert_exceptions


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationException("OpenAI API key is required")
            
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            return AsyncOpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using OpenAI."""
        try:
            model = self.config.get("model_name", "gpt-4o")
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.0))
            max_tokens = kwargs.get("max_tokens", 4000)
            response_format = kwargs.get("response_format")
            
            # Remove temperature, max_tokens, and response_format from kwargs to avoid duplicate arguments
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "response_format"]}
            
            # Check if response_format is a BaseModel - if so, use parse() instead of create()
            from pydantic import BaseModel
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                response = await self.client.chat.completions.parse(
                    model=model,
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
                    "model": model,
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
            logger.error(f"OpenAI chat completion failed: {e}")
            raise ProviderException(f"OpenAI chat completion failed: {e}")


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationException("OpenAI API key is required")
            
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            return AsyncOpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using OpenAI."""
        try:
            model = self.config.get("embedding_model", "text-embedding-3-small")
            
            response = await self.client.embeddings.create(
                model=model,
                input=text,
                **kwargs
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise ProviderException(f"OpenAI embedding failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI."""
        try:
            model = self.config.get("embedding_model", "text-embedding-3-small")
            
            response = await self.client.embeddings.create(
                model=model,
                input=texts,
                **kwargs
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise ProviderException(f"OpenAI batch embedding failed: {e}")


class OpenAITranscriptionProvider(TranscriptionProvider):
    """OpenAI Whisper transcription provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationException("OpenAI API key is required")
            
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            return AsyncOpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """Transcribe audio bytes using OpenAI Whisper."""
        try:
            model = self.config.get("model", "whisper-1")
            
            # Create a temporary file-like object
            import io
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"  # Whisper needs a filename
            
            response = await self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language,
                **kwargs
            )
            
            return response.text
        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            raise ProviderException(f"OpenAI Whisper transcription failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio file using OpenAI Whisper."""
        try:
            model = self.config.get("model", "whisper-1")
            
            with open(audio_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language,
                    **kwargs
                )
            
            return response.text
        except Exception as e:
            logger.error(f"OpenAI Whisper file transcription failed: {e}")
            raise ProviderException(f"OpenAI Whisper file transcription failed: {e}")


class OpenAIVisionProvider(VisionProvider):
    """OpenAI Vision provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationException("OpenAI API key is required")
            
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            return AsyncOpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def analyze_image(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """Analyze image using OpenAI Vision."""
        try:
            model = self.config.get("model", "gpt-4o")
            prompt = kwargs.get("prompt", "Analyze this image and describe what you see.")
            
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.0)
            )
            
            return {
                "analysis": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None
            }
        except Exception as e:
            logger.error(f"OpenAI Vision analysis failed: {e}")
            raise ProviderException(f"OpenAI Vision analysis failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def extract_text(self, image_data: bytes, **kwargs) -> str:
        """Extract text from image using OpenAI Vision."""
        try:
            prompt = "Extract all text from this image and return it exactly as it appears."
            
            result = await self.analyze_image(image_data, prompt=prompt, **kwargs)
            return result["analysis"]
        except Exception as e:
            logger.error(f"OpenAI Vision text extraction failed: {e}")
            raise ProviderException(f"OpenAI Vision text extraction failed: {e}")