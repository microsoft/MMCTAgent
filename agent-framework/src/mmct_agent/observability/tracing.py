"""Tracing utilities for request tracking."""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


# Context variable for trace ID
_current_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


@dataclass
class Span:
    """A span representing a unit of work in a trace."""
    
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid4())[:8])
    parent_span_id: str | None = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # "ok", "error"
    error_message: str | None = None
    
    @property
    def duration_ms(self) -> float | None:
        """Calculate span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute.
        
        Args:
            key: Attribute key.
            value: Attribute value.
        """
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span.
        
        Args:
            name: Event name.
            attributes: Event attributes.
        """
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })
    
    def set_error(self, message: str) -> None:
        """Mark span as error.
        
        Args:
            message: Error message.
        """
        self.status = "error"
        self.error_message = message
    
    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.utcnow()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "error_message": self.error_message,
        }


class TraceContext:
    """Context manager for tracing operations."""
    
    def __init__(self, trace_id: str | None = None) -> None:
        """Initialize trace context.
        
        Args:
            trace_id: Optional trace ID. Generated if not provided.
        """
        self.trace_id = trace_id or str(uuid4())
        self._spans: list[Span] = []
        self._token: Any = None
    
    def __enter__(self) -> TraceContext:
        """Enter the trace context."""
        self._token = _current_trace_id.set(self.trace_id)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the trace context."""
        if self._token:
            _current_trace_id.reset(self._token)
    
    async def __aenter__(self) -> TraceContext:
        """Async enter the trace context."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async exit the trace context."""
        self.__exit__(exc_type, exc_val, exc_tb)
    
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.
        
        Args:
            name: Span name.
            attributes: Initial attributes.
            
        Returns:
            New Span instance.
        """
        parent_span = _current_span.get()
        parent_id = parent_span.span_id if parent_span else None
        
        span = Span(
            name=name,
            trace_id=self.trace_id,
            parent_span_id=parent_id,
            attributes=attributes or {},
        )
        
        self._spans.append(span)
        _current_span.set(span)
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End a span.
        
        Args:
            span: Span to end.
        """
        span.end()
        
        # Restore parent span
        if span.parent_span_id:
            for s in reversed(self._spans):
                if s.span_id == span.parent_span_id:
                    _current_span.set(s)
                    break
        else:
            _current_span.set(None)
    
    @property
    def spans(self) -> list[Span]:
        """Get all spans in this trace."""
        return self._spans.copy()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self._spans],
        }


def get_current_trace_id() -> str | None:
    """Get the current trace ID.
    
    Returns:
        Current trace ID or None.
    """
    return _current_trace_id.get()


def get_current_span() -> Span | None:
    """Get the current span.
    
    Returns:
        Current span or None.
    """
    return _current_span.get()


class SpanContext:
    """Context manager for a single span."""
    
    def __init__(
        self,
        name: str,
        trace_context: TraceContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Initialize span context.
        
        Args:
            name: Span name.
            trace_context: Parent trace context.
            attributes: Initial attributes.
        """
        self.name = name
        self.trace_context = trace_context
        self.attributes = attributes or {}
        self.span: Span | None = None
        self._token: Any = None
    
    def __enter__(self) -> Span:
        """Enter the span context."""
        trace_id = get_current_trace_id() or str(uuid4())
        
        parent_span = _current_span.get()
        parent_id = parent_span.span_id if parent_span else None
        
        self.span = Span(
            name=self.name,
            trace_id=trace_id,
            parent_span_id=parent_id,
            attributes=self.attributes,
        )
        
        self._token = _current_span.set(self.span)
        return self.span
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the span context."""
        if self.span:
            if exc_val:
                self.span.set_error(str(exc_val))
            self.span.end()
        
        if self._token:
            _current_span.reset(self._token)
    
    async def __aenter__(self) -> Span:
        """Async enter."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async exit."""
        self.__exit__(exc_type, exc_val, exc_tb)
