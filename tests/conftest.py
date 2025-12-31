"""Pytest configuration and fixtures for image-gen-mcp tests."""

import base64
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Sample base64-encoded 1x1 PNG image (valid PNG)
SAMPLE_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# Sample base64-encoded 1x1 JPEG image (valid JPEG)
SAMPLE_JPEG_BASE64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDAREAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/wA//2Q=="


@pytest.fixture
def sample_png_bytes():
    """Return valid PNG image bytes."""
    return base64.b64decode(SAMPLE_PNG_BASE64)


@pytest.fixture
def sample_jpeg_bytes():
    """Return valid JPEG image bytes."""
    return base64.b64decode(SAMPLE_JPEG_BASE64)


@pytest.fixture
def sample_png_base64():
    """Return base64-encoded PNG."""
    return SAMPLE_PNG_BASE64


@pytest.fixture
def sample_jpeg_base64():
    """Return base64-encoded JPEG."""
    return SAMPLE_JPEG_BASE64


@pytest.fixture
def mock_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "generated-images"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def mock_provider():
    """Create a mock image provider."""
    from image_gen_mcp.providers.base import (
        ImageProvider,
        GenerationResult,
        ProviderModel,
        ProviderStatus,
        ImageFormat,
    )

    class MockProvider(ImageProvider):
        name = "mock"
        display_name = "Mock Provider"
        requires_api_key = False
        free_tier = True

        async def generate(
            self,
            prompt: str,
            negative_prompt=None,
            width=1024,
            height=1024,
            seed=None,
            model=None,
            **kwargs
        ):
            return GenerationResult(
                success=True,
                provider=self.name,
                model=model or "mock-model",
                prompt=prompt,
                image_base64=SAMPLE_PNG_BASE64,
                image_format=ImageFormat.PNG,
                mime_type="image/png",
                width=width,
                height=height,
                seed=seed,
                generation_time_ms=100,
                cost=0.0,
            )

        async def list_models(self):
            return [
                ProviderModel(
                    id="mock-model",
                    name="Mock Model",
                    description="Test model",
                    max_width=1024,
                    max_height=1024,
                    cost_per_image=0.0,
                )
            ]

    return MockProvider()


@pytest.fixture
def mock_aiohttp_response():
    """Create mock aiohttp response."""
    async def _create_response(status=200, data=None, text=""):
        response = AsyncMock()
        response.status = status
        response.read = AsyncMock(return_value=data or base64.b64decode(SAMPLE_PNG_BASE64))
        response.text = AsyncMock(return_value=text)
        return response
    return _create_response


@pytest.fixture
def mock_aiohttp_session(mock_aiohttp_response):
    """Create mock aiohttp ClientSession."""
    async def _create_session(status=200, data=None):
        session = MagicMock()

        async def _create_response_cm(*args, **kwargs):
            response = await mock_aiohttp_response(status=status, data=data)
            return response

        # Create async context manager
        class AsyncContextManager:
            async def __aenter__(self):
                return await _create_response_cm()

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        session.get = MagicMock(return_value=AsyncContextManager())
        session.head = MagicMock(return_value=AsyncContextManager())

        return session
    return _create_session
