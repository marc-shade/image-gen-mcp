"""Simplified tests for Pollinations.ai provider."""

import pytest
from image_gen_mcp.providers.pollinations import PollinationsProvider
from image_gen_mcp.providers.base import ImageFormat, ProviderStatus


class TestPollinationsProvider:
    """Test Pollinations.ai provider."""

    @pytest.fixture
    def provider(self):
        """Create Pollinations provider instance."""
        return PollinationsProvider()

    def test_provider_initialization(self, provider):
        """Test provider is properly initialized."""
        assert provider.name == "pollinations"
        assert provider.display_name == "Pollinations.ai (FREE)"
        assert provider.requires_api_key is False
        assert provider.free_tier is True
        assert provider.BASE_URL == "https://image.pollinations.ai/prompt"

    def test_provider_status(self, provider):
        """Test provider status is available (no API key needed)."""
        assert provider.status == ProviderStatus.AVAILABLE

    def test_available_models(self, provider):
        """Test provider has expected models."""
        expected_models = ["flux", "flux-realism", "flux-anime", "flux-3d", "turbo"]
        assert all(model in provider.MODELS for model in expected_models)

    def test_model_configuration(self, provider):
        """Test model configurations are valid."""
        flux_model = provider.MODELS["flux"]
        assert flux_model.id == "flux"
        assert flux_model.name == "Flux"
        assert flux_model.max_width == 1440
        assert flux_model.max_height == 1440
        assert flux_model.supports_seed is True
        assert flux_model.cost_per_image == 0.0

        turbo_model = provider.MODELS["turbo"]
        assert turbo_model.id == "turbo"
        assert turbo_model.max_width == 1024
        assert turbo_model.average_generation_time_ms < flux_model.average_generation_time_ms

    @pytest.mark.asyncio
    async def test_list_models(self, provider):
        """Test listing available models."""
        models = await provider.list_models()
        assert len(models) == len(provider.MODELS)
        assert all(hasattr(m, 'id') for m in models)
        assert all(hasattr(m, 'name') for m in models)

    def test_request_tracking(self, provider):
        """Test that requests are tracked."""
        initial_count = provider._request_count

        provider._record_request(0.0)

        assert provider._request_count == initial_count + 1
        assert provider._total_cost == 0.0  # Pollinations is free
