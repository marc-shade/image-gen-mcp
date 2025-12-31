"""Tests for base provider functionality."""

import pytest
from image_gen_mcp.providers.base import (
    ImageFormat,
    detect_image_format,
    convert_to_png,
    ProviderStatus,
    GenerationResult,
    ProviderModel,
    ImageProvider,
)


class TestImageFormat:
    """Test ImageFormat enum."""

    def test_image_format_extensions(self):
        """Test file extensions for each format."""
        assert ImageFormat.PNG.extension == ".png"
        assert ImageFormat.JPEG.extension == ".jpg"
        assert ImageFormat.WEBP.extension == ".webp"
        assert ImageFormat.GIF.extension == ".gif"
        assert ImageFormat.UNKNOWN.extension == ".bin"

    def test_image_format_mime_types(self):
        """Test MIME types for each format."""
        assert ImageFormat.PNG.mime_type == "image/png"
        assert ImageFormat.JPEG.mime_type == "image/jpeg"
        assert ImageFormat.WEBP.mime_type == "image/webp"
        assert ImageFormat.GIF.mime_type == "image/gif"
        assert ImageFormat.UNKNOWN.mime_type == "application/octet-stream"


class TestDetectImageFormat:
    """Test image format detection from magic bytes."""

    def test_detect_png_format(self, sample_png_bytes):
        """Test PNG format detection."""
        assert detect_image_format(sample_png_bytes) == ImageFormat.PNG

    def test_detect_jpeg_format(self, sample_jpeg_bytes):
        """Test JPEG format detection."""
        assert detect_image_format(sample_jpeg_bytes) == ImageFormat.JPEG

    def test_detect_webp_format(self):
        """Test WebP format detection."""
        # WebP magic bytes: RIFF....WEBP
        webp_bytes = b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 10
        assert detect_image_format(webp_bytes) == ImageFormat.WEBP

    def test_detect_gif_format(self):
        """Test GIF format detection."""
        gif87_bytes = b'GIF87a' + b'\x00' * 10
        gif89_bytes = b'GIF89a' + b'\x00' * 10
        assert detect_image_format(gif87_bytes) == ImageFormat.GIF
        assert detect_image_format(gif89_bytes) == ImageFormat.GIF

    def test_detect_unknown_format(self):
        """Test unknown format detection."""
        unknown_bytes = b'INVALID' + b'\x00' * 10
        assert detect_image_format(unknown_bytes) == ImageFormat.UNKNOWN

    def test_detect_format_insufficient_data(self):
        """Test format detection with insufficient data."""
        assert detect_image_format(b'abc') == ImageFormat.UNKNOWN
        assert detect_image_format(b'') == ImageFormat.UNKNOWN


class TestConvertToPng:
    """Test PNG conversion functionality."""

    def test_convert_jpeg_to_png(self, sample_jpeg_bytes):
        """Test converting JPEG to PNG."""
        result = convert_to_png(sample_jpeg_bytes)
        assert isinstance(result, bytes)
        # Should return PNG format
        assert detect_image_format(result) in (ImageFormat.PNG, ImageFormat.JPEG)

    def test_convert_invalid_data(self):
        """Test conversion with invalid image data."""
        invalid_data = b'not an image'
        result = convert_to_png(invalid_data)
        # Should return original data on failure
        assert result == invalid_data


class TestProviderStatus:
    """Test ProviderStatus enum."""

    def test_provider_status_values(self):
        """Test all provider status values exist."""
        assert ProviderStatus.AVAILABLE.value == "available"
        assert ProviderStatus.RATE_LIMITED.value == "rate_limited"
        assert ProviderStatus.ERROR.value == "error"
        assert ProviderStatus.UNCONFIGURED.value == "unconfigured"
        assert ProviderStatus.DISABLED.value == "disabled"


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_generation_result_success(self, sample_png_base64):
        """Test successful generation result."""
        result = GenerationResult(
            success=True,
            provider="test",
            model="test-model",
            prompt="test prompt",
            image_base64=sample_png_base64,
            image_format=ImageFormat.PNG,
            mime_type="image/png",
            width=1024,
            height=1024,
            seed=42,
            generation_time_ms=5000,
            cost=0.003,
        )

        assert result.success is True
        assert result.provider == "test"
        assert result.model == "test-model"
        assert result.prompt == "test prompt"
        assert result.image_base64 == sample_png_base64
        assert result.width == 1024
        assert result.height == 1024
        assert result.seed == 42
        assert result.error is None

    def test_generation_result_failure(self):
        """Test failed generation result."""
        result = GenerationResult(
            success=False,
            provider="test",
            model="test-model",
            prompt="test prompt",
            error="Test error message",
        )

        assert result.success is False
        assert result.error == "Test error message"
        assert result.image_base64 is None

    def test_generation_result_to_dict(self, sample_png_base64):
        """Test converting result to dictionary."""
        result = GenerationResult(
            success=True,
            provider="test",
            model="test-model",
            prompt="test prompt",
            image_base64=sample_png_base64,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["provider"] == "test"
        assert result_dict["prompt"] == "test prompt"
        # Base64 should be truncated in dict
        assert "..." in result_dict["image_base64"]


class TestProviderModel:
    """Test ProviderModel dataclass."""

    def test_provider_model_defaults(self):
        """Test ProviderModel with default values."""
        model = ProviderModel(
            id="test-model",
            name="Test Model",
            description="A test model",
        )

        assert model.id == "test-model"
        assert model.name == "Test Model"
        assert model.max_width == 1024
        assert model.max_height == 1024
        assert model.supports_negative_prompt is True
        assert model.supports_seed is True
        assert model.supports_img2img is False
        assert model.cost_per_image == 0.0

    def test_provider_model_custom_values(self):
        """Test ProviderModel with custom values."""
        model = ProviderModel(
            id="custom-model",
            name="Custom Model",
            description="Custom test",
            max_width=2048,
            max_height=2048,
            supports_negative_prompt=False,
            cost_per_image=0.01,
            average_generation_time_ms=10000,
        )

        assert model.max_width == 2048
        assert model.max_height == 2048
        assert model.supports_negative_prompt is False
        assert model.cost_per_image == 0.01
        assert model.average_generation_time_ms == 10000


class TestImageProvider:
    """Test ImageProvider base class."""

    def test_provider_initialization_no_key(self, mock_provider):
        """Test provider initialization without API key."""
        assert mock_provider.name == "mock"
        assert mock_provider.display_name == "Mock Provider"
        assert mock_provider.requires_api_key is False
        assert mock_provider.free_tier is True
        assert mock_provider._request_count == 0
        assert mock_provider._total_cost == 0.0

    def test_provider_initialization_with_key(self):
        """Test provider initialization with API key."""
        from image_gen_mcp.providers.base import ImageProvider

        class TestProvider(ImageProvider):
            name = "test"
            requires_api_key = True

            async def generate(self, *args, **kwargs):
                pass

            async def list_models(self):
                return []

        provider = TestProvider(api_key="test-key-123")
        assert provider.api_key == "test-key-123"

    def test_provider_status_unconfigured(self):
        """Test provider status when API key required but not provided."""
        from image_gen_mcp.providers.base import ImageProvider

        class TestProvider(ImageProvider):
            name = "test"
            requires_api_key = True

            async def generate(self, *args, **kwargs):
                pass

            async def list_models(self):
                return []

        provider = TestProvider()  # No API key
        assert provider.status == ProviderStatus.UNCONFIGURED

    def test_provider_status_configured(self):
        """Test provider status when properly configured."""
        from image_gen_mcp.providers.base import ImageProvider

        class TestProvider(ImageProvider):
            name = "test"
            requires_api_key = True

            async def generate(self, *args, **kwargs):
                pass

            async def list_models(self):
                return []

        provider = TestProvider(api_key="test-key")
        assert provider.status == ProviderStatus.AVAILABLE

    def test_provider_record_request(self, mock_provider):
        """Test request recording."""
        initial_count = mock_provider._request_count
        initial_cost = mock_provider._total_cost

        mock_provider._record_request(cost=0.05)

        assert mock_provider._request_count == initial_count + 1
        assert mock_provider._total_cost == initial_cost + 0.05

    def test_provider_encode_image(self, mock_provider, sample_png_bytes):
        """Test base64 image encoding."""
        import base64

        encoded = mock_provider._encode_image(sample_png_bytes)
        assert isinstance(encoded, str)

        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == sample_png_bytes

    @pytest.mark.asyncio
    async def test_provider_check_health(self, mock_provider):
        """Test provider health check."""
        health = await mock_provider.check_health()

        assert isinstance(health, dict)
        assert health["provider"] == "mock"
        assert health["status"] == "available"
        assert health["configured"] is True
        assert "request_count" in health
        assert "total_cost" in health

    @pytest.mark.asyncio
    async def test_provider_generate(self, mock_provider):
        """Test image generation through mock provider."""
        result = await mock_provider.generate(
            prompt="test prompt",
            width=512,
            height=512,
            seed=42,
        )

        assert result.success is True
        assert result.provider == "mock"
        assert result.prompt == "test prompt"
        assert result.width == 512
        assert result.height == 512
        assert result.seed == 42

    @pytest.mark.asyncio
    async def test_provider_list_models(self, mock_provider):
        """Test listing provider models."""
        models = await mock_provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, ProviderModel) for m in models)
