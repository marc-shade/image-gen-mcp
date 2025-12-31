"""Tests for MCP server functionality."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import mcp.types as types
from image_gen_mcp.server import (
    generate_image,
    generate_pixel_art,
    list_providers,
    list_models,
    get_provider_status,
    save_image,
    init_providers,
    PROVIDERS,
)
from image_gen_mcp.providers.base import (
    GenerationResult,
    ImageFormat,
    ProviderStatus,
)


class TestInitProviders:
    """Test provider initialization."""

    def test_providers_initialized(self):
        """Test that all providers are initialized."""
        init_providers()

        assert "pollinations" in PROVIDERS
        assert "cloudflare" in PROVIDERS
        assert "huggingface" in PROVIDERS
        assert "together" in PROVIDERS
        assert "replicate" in PROVIDERS

    def test_providers_are_instances(self):
        """Test that providers are proper instances."""
        from image_gen_mcp.providers.base import ImageProvider

        for name, provider in PROVIDERS.items():
            assert isinstance(provider, ImageProvider)
            assert hasattr(provider, 'generate')
            assert hasattr(provider, 'list_models')


class TestGenerateImage:
    """Test generate_image tool."""

    @pytest.mark.asyncio
    async def test_generate_image_success(self, mock_provider, sample_png_base64):
        """Test successful image generation."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': 'a beautiful landscape',
                'width': 1024,
                'height': 1024,
                'provider': 'test',
                'save_to_file': False,
            })

            assert len(result) == 2
            assert isinstance(result[0], types.ImageContent)
            assert isinstance(result[1], types.TextContent)

            # Check image content
            assert result[0].type == "image"
            assert result[0].data == sample_png_base64
            assert result[0].mimeType == "image/png"

            # Check metadata
            metadata = json.loads(result[1].text)
            assert metadata['success'] is True
            assert metadata['provider'] == 'mock'
            assert metadata['prompt'] == 'a beautiful landscape'
            assert metadata['width'] == 1024
            assert metadata['height'] == 1024

    @pytest.mark.asyncio
    async def test_generate_image_empty_prompt(self):
        """Test generation with empty prompt."""
        result = await generate_image({'prompt': ''})

        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)

        response = json.loads(result[0].text)
        assert response['success'] is False
        assert 'required' in response['error'].lower()

    @pytest.mark.asyncio
    async def test_generate_image_missing_prompt(self):
        """Test generation with missing prompt."""
        result = await generate_image({})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response['success'] is False

    @pytest.mark.asyncio
    async def test_generate_image_provider_fallback(self, mock_provider):
        """Test provider fallback when first provider fails."""
        failing_provider = AsyncMock()
        failing_provider.status = ProviderStatus.AVAILABLE
        failing_provider.generate = AsyncMock(return_value=GenerationResult(
            success=False,
            provider="failing",
            model="test",
            prompt="test",
            error="Failed to generate",
        ))

        providers = {
            'failing': failing_provider,
            'working': mock_provider,
        }

        with patch.dict('image_gen_mcp.server.PROVIDERS', providers):
            # Don't specify provider, should try fallback
            with patch('image_gen_mcp.server.provider_order', ['failing', 'working']):
                result = await generate_image({
                    'prompt': 'test',
                    'save_to_file': False,
                })

                metadata = json.loads(result[1].text)
                # Should succeed with working provider
                assert metadata['success'] is True
                assert metadata['provider'] == 'mock'

    @pytest.mark.asyncio
    async def test_generate_image_all_providers_fail(self):
        """Test when all providers fail."""
        failing_provider = AsyncMock()
        failing_provider.status = ProviderStatus.AVAILABLE
        failing_provider.generate = AsyncMock(return_value=GenerationResult(
            success=False,
            provider="failing",
            model="test",
            prompt="test",
            error="All failed",
        ))

        with patch.dict('image_gen_mcp.server.PROVIDERS', {'failing': failing_provider}):
            result = await generate_image({
                'prompt': 'test',
                'save_to_file': False,
            })

            assert len(result) == 1
            response = json.loads(result[0].text)
            assert response['success'] is False
            assert 'error' in response

    @pytest.mark.asyncio
    async def test_generate_image_unconfigured_provider(self):
        """Test skipping unconfigured providers."""
        unconfigured_provider = AsyncMock()
        unconfigured_provider.status = ProviderStatus.UNCONFIGURED

        with patch.dict('image_gen_mcp.server.PROVIDERS', {'unconfigured': unconfigured_provider}):
            result = await generate_image({
                'prompt': 'test',
                'provider': 'unconfigured',
                'save_to_file': False,
            })

            response = json.loads(result[0].text)
            assert response['success'] is False

    @pytest.mark.asyncio
    async def test_generate_image_with_seed(self, mock_provider):
        """Test generation with specific seed."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': 'test',
                'seed': 42,
                'provider': 'test',
                'save_to_file': False,
            })

            metadata = json.loads(result[1].text)
            assert metadata['seed'] == 42

    @pytest.mark.asyncio
    async def test_generate_image_with_model(self, mock_provider):
        """Test generation with specific model."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': 'test',
                'model': 'custom-model',
                'provider': 'test',
                'save_to_file': False,
            })

            metadata = json.loads(result[1].text)
            assert metadata['model'] in ['custom-model', 'mock-model']

    @pytest.mark.asyncio
    async def test_generate_image_custom_dimensions(self, mock_provider):
        """Test generation with custom dimensions."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': 'test',
                'width': 512,
                'height': 768,
                'provider': 'test',
                'save_to_file': False,
            })

            metadata = json.loads(result[1].text)
            assert metadata['width'] == 512
            assert metadata['height'] == 768

    @pytest.mark.asyncio
    async def test_generate_image_file_saving(self, mock_provider, mock_output_dir):
        """Test image file saving."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
                result = await generate_image({
                    'prompt': 'test prompt',
                    'provider': 'test',
                    'save_to_file': True,
                })

                metadata = json.loads(result[1].text)
                assert metadata['file_path'] is not None

                # Check file was created
                file_path = Path(metadata['file_path'])
                assert file_path.exists()
                assert file_path.suffix == '.png'


class TestGeneratePixelArt:
    """Test generate_pixel_art tool."""

    @pytest.mark.asyncio
    async def test_generate_pixel_art_success(self, mock_provider):
        """Test successful pixel art generation."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'huggingface': mock_provider}):
            result = await generate_pixel_art({
                'prompt': 'dragon sprite',
                'style': 'nes',
                'size': 64,
                'colors': 16,
            })

            assert len(result) == 2
            assert isinstance(result[0], types.ImageContent)

            metadata = json.loads(result[1].text)
            assert metadata['success'] is True
            assert metadata['type'] == 'pixel_art'
            assert metadata['style'] == 'nes'
            assert metadata['size'] == 64

    @pytest.mark.asyncio
    async def test_generate_pixel_art_empty_prompt(self):
        """Test pixel art with empty prompt."""
        result = await generate_pixel_art({'prompt': ''})

        response = json.loads(result[0].text)
        assert response['success'] is False

    @pytest.mark.asyncio
    async def test_generate_pixel_art_styles(self, mock_provider):
        """Test different pixel art styles."""
        styles = ['nes', 'snes', 'gameboy', 'modern', 'isometric']

        for style in styles:
            with patch.dict('image_gen_mcp.server.PROVIDERS', {'huggingface': mock_provider}):
                result = await generate_pixel_art({
                    'prompt': 'character',
                    'style': style,
                })

                metadata = json.loads(result[1].text)
                assert metadata['style'] == style

    @pytest.mark.asyncio
    async def test_generate_pixel_art_sizes(self, mock_provider):
        """Test different pixel art sizes."""
        sizes = [64, 128, 256, 512]

        for size in sizes:
            with patch.dict('image_gen_mcp.server.PROVIDERS', {'huggingface': mock_provider}):
                result = await generate_pixel_art({
                    'prompt': 'sprite',
                    'size': size,
                })

                metadata = json.loads(result[1].text)
                assert metadata['size'] == size

    @pytest.mark.asyncio
    async def test_generate_pixel_art_provider_fallback(self, mock_provider):
        """Test pixel art provider fallback."""
        failing_provider = AsyncMock()
        failing_provider.status = ProviderStatus.AVAILABLE
        failing_provider.generate = AsyncMock(return_value=GenerationResult(
            success=False,
            provider="failing",
            model="test",
            prompt="test",
            error="Failed",
        ))

        providers = {
            'huggingface': failing_provider,
            'pollinations': mock_provider,
        }

        with patch.dict('image_gen_mcp.server.PROVIDERS', providers):
            result = await generate_pixel_art({'prompt': 'test'})

            metadata = json.loads(result[1].text)
            assert metadata['success'] is True


class TestListProviders:
    """Test list_providers tool."""

    @pytest.mark.asyncio
    async def test_list_providers(self):
        """Test listing all providers."""
        result = await list_providers({})

        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)

        data = json.loads(result[0].text)
        assert 'providers' in data
        assert 'default' in data
        assert 'fallback_order' in data

        # Check provider info
        providers = data['providers']
        assert len(providers) > 0

        for provider in providers:
            assert 'name' in provider
            assert 'display_name' in provider
            assert 'status' in provider
            assert 'requires_api_key' in provider
            assert 'free_tier' in provider


class TestListModels:
    """Test list_models tool."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, mock_provider):
        """Test listing models for a provider."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await list_models({'provider': 'test'})

            data = json.loads(result[0].text)
            assert data['provider'] == 'test'
            assert 'models' in data
            assert len(data['models']) > 0

            model = data['models'][0]
            assert 'id' in model
            assert 'name' in model
            assert 'description' in model

    @pytest.mark.asyncio
    async def test_list_models_unknown_provider(self):
        """Test listing models for unknown provider."""
        result = await list_models({'provider': 'nonexistent'})

        data = json.loads(result[0].text)
        assert data['success'] is False
        assert 'error' in data


class TestGetProviderStatus:
    """Test get_provider_status tool."""

    @pytest.mark.asyncio
    async def test_get_provider_status_success(self, mock_provider):
        """Test getting provider status."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await get_provider_status({'provider': 'test'})

            data = json.loads(result[0].text)
            assert 'provider' in data
            assert 'status' in data
            assert 'configured' in data

    @pytest.mark.asyncio
    async def test_get_provider_status_unknown(self):
        """Test status for unknown provider."""
        result = await get_provider_status({'provider': 'nonexistent'})

        data = json.loads(result[0].text)
        assert data['success'] is False


class TestSaveImage:
    """Test save_image tool."""

    @pytest.mark.asyncio
    async def test_save_image_success(self, sample_png_base64, mock_output_dir):
        """Test saving image successfully."""
        with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
            result = await save_image({
                'image_base64': sample_png_base64,
                'filename': 'test_image',
                'format': 'png',
            })

            data = json.loads(result[0].text)
            assert data['success'] is True
            assert 'file_path' in data
            assert 'size_bytes' in data

            # Verify file exists
            file_path = Path(data['file_path'])
            assert file_path.exists()
            assert file_path.name == 'test_image.png'

    @pytest.mark.asyncio
    async def test_save_image_empty_data(self):
        """Test saving with empty base64 data."""
        result = await save_image({
            'image_base64': '',
            'filename': 'test',
        })

        data = json.loads(result[0].text)
        assert data['success'] is False
        assert 'error' in data

    @pytest.mark.asyncio
    async def test_save_image_different_formats(self, sample_png_base64, mock_output_dir):
        """Test saving in different formats."""
        formats = ['png', 'jpg', 'webp']

        for fmt in formats:
            with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
                result = await save_image({
                    'image_base64': sample_png_base64,
                    'filename': f'test_{fmt}',
                    'format': fmt,
                })

                data = json.loads(result[0].text)
                assert data['success'] is True

    @pytest.mark.asyncio
    async def test_save_image_invalid_base64(self, mock_output_dir):
        """Test saving with invalid base64 data."""
        with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
            result = await save_image({
                'image_base64': 'not-valid-base64!@#$',
                'filename': 'test',
            })

            data = json.loads(result[0].text)
            assert data['success'] is False

    @pytest.mark.asyncio
    async def test_save_image_auto_filename(self, sample_png_base64, mock_output_dir):
        """Test saving with auto-generated filename."""
        with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
            result = await save_image({
                'image_base64': sample_png_base64,
            })

            data = json.loads(result[0].text)
            if data['success']:
                assert 'file_path' in data
                # Should have timestamp-based filename
                file_path = Path(data['file_path'])
                assert file_path.exists()
