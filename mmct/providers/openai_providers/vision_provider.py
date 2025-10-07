from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.exceptions import ProviderException
from mmct.providers.base import VisionProvider
from openai import AsyncOpenAI, OpenAI
from mmct.exceptions import ProviderException, ConfigurationException
from loguru import logger
from typing import Dict, Any

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