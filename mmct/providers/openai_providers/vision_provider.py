from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.providers.base import BaseVisionProvider
from openai import AsyncOpenAI, OpenAI
from mmct.utils.error_handler import ProviderException, ConfigurationException
from loguru import logger
from typing import Dict, Any, Optional

class OpenAIVisionProvider(BaseVisionProvider):
    """OpenAI Vision provider implementation."""
    
    def __init__(self, api_key:str, model_name:str, timeout:Optional[int] = 200, max_retries:Optional[int] = 2):
        if not api_key:
                raise ConfigurationException("OpenAI API key is required!")
        
        if not model_name:
            raise ConfigurationException("OpenAI model name is required!")

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_name = model_name
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            return AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def analyze_image(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """Analyze image using OpenAI Vision."""
        try:
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
                model=self.model_name,
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