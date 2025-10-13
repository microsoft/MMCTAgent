import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, validator, Field
from ..utils.error_handler import ValidationException


class QueryRequest(BaseModel):
    """Request model for query operations."""
    
    query: str = Field(..., min_length=1, max_length=1000)
    max_results: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(default=None)
    include_metadata: bool = Field(default=True)
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValidationException('Query cannot be empty')
        
        # Remove potentially malicious content
        sanitized = re.sub(r'[<>"\']', '', v)
        sanitized = sanitized.strip()
        
        if len(sanitized) == 0:
            raise ValidationException('Query contains only invalid characters')
            
        return sanitized


class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis operations."""
    
    image_path: str = Field(..., min_length=1)
    tools: List[str] = Field(..., min_items=1)
    use_critic: bool = Field(default=True)
    timeout: int = Field(default=30, ge=1, le=300)
    
    @validator('image_path')
    def validate_image_path(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValidationException('Image path cannot be empty')
        
        # Basic path validation
        if '..' in v or v.startswith('/'):
            raise ValidationException('Invalid image path')
            
        return v.strip()
    
    @validator('tools')
    def validate_tools(cls, v):
        valid_tools = ['ocr', 'object_detection', 'recognition', 'vision', 'critic']
        for tool in v:
            if tool not in valid_tools:
                raise ValidationException(f'Invalid tool: {tool}. Valid tools: {valid_tools}')
        return v


class VideoAnalysisRequest(BaseModel):
    """Request model for video analysis operations."""
    
    video_path: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=1000)
    frame_interval: int = Field(default=30, ge=1, le=300)
    max_frames: int = Field(default=100, ge=1, le=1000)
    
    @validator('video_path')
    def validate_video_path(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValidationException('Video path cannot be empty')
        
        # Basic path validation
        if '..' in v or v.startswith('/'):
            raise ValidationException('Invalid video path')
            
        return v.strip()
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValidationException('Query cannot be empty')
        
        # Remove potentially malicious content
        sanitized = re.sub(r'[<>"\']', '', v)
        sanitized = sanitized.strip()
        
        if len(sanitized) == 0:
            raise ValidationException('Query contains only invalid characters')
            
        return sanitized


class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion operations."""
    
    document_path: str = Field(..., min_length=1)
    index_name: str = Field(..., min_length=1, max_length=100)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    overlap: int = Field(default=100, ge=0, le=1000)
    
    @validator('document_path')
    def validate_document_path(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValidationException('Document path cannot be empty')
        
        # Basic path validation
        if '..' in v:
            raise ValidationException('Invalid document path')
            
        return v.strip()
    
    @validator('index_name')
    def validate_index_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValidationException('Index name cannot be empty')
        
        # Only allow alphanumeric characters and hyphens
        if not re.match(r'^[a-zA-Z0-9-]+$', v):
            raise ValidationException('Index name can only contain alphanumeric characters and hyphens')
            
        return v.strip().lower()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal attacks."""
    if not filename:
        raise ValidationException('Filename cannot be empty')
    
    # Remove directory traversal attempts
    filename = filename.replace('..', '')
    filename = filename.replace('/', '')
    filename = filename.replace('\\', '')
    
    # Remove potentially dangerous characters
    filename = re.sub(r'[<>:"|?*]', '', filename)
    
    if not filename:
        raise ValidationException('Filename contains only invalid characters')
    
    return filename.strip()


def validate_file_size(file_size: int, max_size: int = 100 * 1024 * 1024) -> bool:
    """Validate file size (default max 100MB)."""
    if file_size <= 0:
        raise ValidationException('File size must be greater than 0')
    
    if file_size > max_size:
        raise ValidationException(f'File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)')
    
    return True


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension."""
    if not filename:
        raise ValidationException('Filename cannot be empty')
    
    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if extension not in [ext.lower() for ext in allowed_extensions]:
        raise ValidationException(f'File extension .{extension} not allowed. Allowed extensions: {allowed_extensions}')
    
    return True