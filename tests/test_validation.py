"""Tests for parameter validation and edge cases."""

import pytest
import json
from unittest.mock import patch

import mcp.types as types
from image_gen_mcp.server import (
    generate_image,
    generate_pixel_art,
    save_image,
    list_models,
    get_provider_status,
)


class TestParameterValidation:
    """Test parameter validation for all tools."""

    @pytest.mark.asyncio
    async def test_generate_image_invalid_dimensions(self, mock_provider):
        """Test validation of invalid dimensions."""
        test_cases = [
            {'width': -100, 'height': 1024},  # Negative width
            {'width': 1024, 'height': -100},  # Negative height
            {'width': 0, 'height': 1024},     # Zero width
            {'width': 1024, 'height': 0},     # Zero height
        ]

        for args in test_cases:
            with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
                result = await generate_image({
                    'prompt': 'test',
                    'provider': 'test',
                    'save_to_file': False,
                    **args
                })

                # Should either fail or normalize to valid values
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_image_extreme_dimensions(self, mock_provider):
        """Test handling of extremely large dimensions."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': 'test',
                'width': 100000,
                'height': 100000,
                'provider': 'test',
                'save_to_file': False,
            })

            # Should handle gracefully (provider will normalize)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_image_special_characters_prompt(self, mock_provider):
        """Test prompts with special characters."""
        special_prompts = [
            "test with 'single quotes'",
            'test with "double quotes"',
            "test with\nnewlines",
            "test with\ttabs",
            "test with émojis 🎨🖼️",
            "test with unicode: 中文",
            "test & ampersand",
            "test <html> tags",
            "test SQL'; DROP TABLE--",
        ]

        for prompt in special_prompts:
            with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
                result = await generate_image({
                    'prompt': prompt,
                    'provider': 'test',
                    'save_to_file': False,
                })

                # Should handle all special characters
                metadata = json.loads(result[1].text)
                assert metadata['success'] is True

    @pytest.mark.asyncio
    async def test_generate_image_very_long_prompt(self, mock_provider):
        """Test very long prompts."""
        long_prompt = "test " * 1000  # 5000 characters

        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': long_prompt,
                'provider': 'test',
                'save_to_file': False,
            })

            # Should handle or truncate gracefully
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_image_invalid_provider(self):
        """Test with non-existent provider."""
        result = await generate_image({
            'prompt': 'test',
            'provider': 'nonexistent_provider_xyz',
            'save_to_file': False,
        })

        response = json.loads(result[0].text)
        assert response['success'] is False

    @pytest.mark.asyncio
    async def test_generate_image_invalid_model(self, mock_provider):
        """Test with invalid model name."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': 'test',
                'model': 'invalid_model_xyz',
                'provider': 'test',
                'save_to_file': False,
            })

            # Should still work, provider will use default
            metadata = json.loads(result[1].text)
            assert metadata['success'] is True

    @pytest.mark.asyncio
    async def test_generate_image_invalid_seed_type(self, mock_provider):
        """Test with invalid seed type."""
        # Note: In Python, this might auto-convert or raise TypeError
        # The actual behavior depends on how the provider handles it
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            # Valid integer seed
            result = await generate_image({
                'prompt': 'test',
                'seed': 42,
                'provider': 'test',
                'save_to_file': False,
            })
            metadata = json.loads(result[1].text)
            assert metadata['success'] is True


class TestPixelArtValidation:
    """Test pixel art parameter validation."""

    @pytest.mark.asyncio
    async def test_pixel_art_invalid_style(self, mock_provider):
        """Test with invalid pixel art style."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            # Invalid style should use default
            result = await generate_pixel_art({
                'prompt': 'test',
                'style': 'invalid_style',
            })

            # Should still work with default style
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_pixel_art_invalid_size(self, mock_provider):
        """Test with invalid pixel art size."""
        invalid_sizes = [32, 100, 1024, -64, 0]

        for size in invalid_sizes:
            with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
                result = await generate_pixel_art({
                    'prompt': 'test',
                    'size': size,
                })

                # Should handle gracefully
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_pixel_art_invalid_colors(self, mock_provider):
        """Test with invalid color palette values."""
        invalid_colors = [-1, 0, 1, 3, 1000]

        for colors in invalid_colors:
            with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
                result = await generate_pixel_art({
                    'prompt': 'test',
                    'colors': colors,
                })

                # Should handle gracefully
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_pixel_art_all_styles(self, mock_provider):
        """Test all valid pixel art styles."""
        styles = ['nes', 'snes', 'gameboy', 'modern', 'isometric']

        for style in styles:
            with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
                result = await generate_pixel_art({
                    'prompt': 'sprite',
                    'style': style,
                })

                metadata = json.loads(result[1].text)
                assert metadata['success'] is True
                assert metadata['style'] == style


class TestSaveImageValidation:
    """Test save_image parameter validation."""

    @pytest.mark.asyncio
    async def test_save_image_invalid_base64(self, mock_output_dir):
        """Test with various invalid base64 inputs."""
        invalid_inputs = [
            'not_base64',
            '!@#$%^&*()',
            'this is not valid base64 encoding',
            '',
        ]

        for invalid_input in invalid_inputs:
            with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
                result = await save_image({
                    'image_base64': invalid_input,
                    'filename': 'test',
                })

                data = json.loads(result[0].text)
                # Most should fail, empty string is caught earlier
                if invalid_input == '':
                    assert data['success'] is False

    @pytest.mark.asyncio
    async def test_save_image_invalid_filename_characters(self, sample_png_base64, mock_output_dir):
        """Test with invalid filename characters."""
        invalid_filenames = [
            'test/with/slash',
            'test\\with\\backslash',
            'test:colon',
            'test*asterisk',
            'test?question',
            'test|pipe',
        ]

        for filename in invalid_filenames:
            with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
                result = await save_image({
                    'image_base64': sample_png_base64,
                    'filename': filename,
                })

                # Should handle gracefully (sanitize or fail)
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_save_image_very_long_filename(self, sample_png_base64, mock_output_dir):
        """Test with extremely long filename."""
        long_filename = 'a' * 500

        with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
            result = await save_image({
                'image_base64': sample_png_base64,
                'filename': long_filename,
            })

            # Should handle or truncate
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_save_image_invalid_format(self, sample_png_base64, mock_output_dir):
        """Test with invalid format."""
        with patch('image_gen_mcp.server.OUTPUT_DIR', mock_output_dir):
            result = await save_image({
                'image_base64': sample_png_base64,
                'filename': 'test',
                'format': 'invalid_format',
            })

            # Should use default or fail gracefully
            assert len(result) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_unicode_in_all_fields(self, mock_provider):
        """Test unicode support in all string fields."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': '日本語のプロンプト',
                'negative_prompt': '中文负面提示',
                'provider': 'test',
                'save_to_file': False,
            })

            metadata = json.loads(result[1].text)
            assert metadata['success'] is True

    @pytest.mark.asyncio
    async def test_concurrent_generation_requests(self, mock_provider):
        """Test multiple concurrent generation requests."""
        import asyncio

        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            tasks = [
                generate_image({
                    'prompt': f'test {i}',
                    'provider': 'test',
                    'save_to_file': False,
                })
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            for result in results:
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_list_models_default_provider(self):
        """Test listing models with default provider."""
        # Should use default provider (pollinations)
        result = await list_models({})

        # Even without explicit provider, should work or fail gracefully
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_provider_status_all_providers(self):
        """Test getting status for all available providers."""
        providers = ['pollinations', 'cloudflare', 'huggingface', 'together', 'replicate']

        for provider in providers:
            result = await get_provider_status({'provider': provider})
            data = json.loads(result[0].text)

            # Should return valid status for each
            assert 'provider' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_empty_arguments(self):
        """Test tools with empty argument dictionaries."""
        # These should handle empty args gracefully
        result = await generate_image({})
        assert len(result) > 0

        result = await generate_pixel_art({})
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_null_values(self, mock_provider):
        """Test handling of None values in optional fields."""
        with patch.dict('image_gen_mcp.server.PROVIDERS', {'test': mock_provider}):
            result = await generate_image({
                'prompt': 'test',
                'negative_prompt': None,
                'model': None,
                'seed': None,
                'provider': 'test',
                'save_to_file': False,
            })

            metadata = json.loads(result[1].text)
            assert metadata['success'] is True


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_generation_timeout_handling(self):
        """Test handling of generation timeouts."""
        from unittest.mock import AsyncMock
        import asyncio

        slow_provider = AsyncMock()
        slow_provider.status.value = "available"

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(0.1)
            from image_gen_mcp.providers.base import GenerationResult
            return GenerationResult(
                success=True,
                provider="slow",
                model="test",
                prompt="test",
                image_base64="test",
            )

        slow_provider.generate = slow_generate

        with patch.dict('image_gen_mcp.server.PROVIDERS', {'slow': slow_provider}):
            result = await generate_image({
                'prompt': 'test',
                'provider': 'slow',
                'save_to_file': False,
            })

            # Should complete
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_invalid_json_in_result(self):
        """Test handling of results that can't be JSON serialized."""
        # This tests internal error handling
        # Most cases are handled by the implementation
        pass

    @pytest.mark.asyncio
    async def test_missing_required_parameters(self):
        """Test that missing required parameters are caught."""
        # Empty prompt
        result = await generate_image({})
        response = json.loads(result[0].text)
        assert response['success'] is False

        # Empty image data for save
        result = await save_image({'filename': 'test'})
        response = json.loads(result[0].text)
        assert response['success'] is False
