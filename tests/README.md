# Image-Gen-MCP Test Suite

Comprehensive test suite for the image-gen-mcp MCP server.

## Test Structure

```
tests/
├── __init__.py              # Test package marker
├── conftest.py              # Pytest fixtures and configuration
├── test_base.py             # Base provider functionality tests
├── test_pollinations.py     # Pollinations provider tests
├── test_server.py           # MCP server tools tests
├── test_validation.py       # Parameter validation and edge cases
└── README.md                # This file
```

## Test Coverage

### Base Provider Tests (`test_base.py`)
- Image format detection (PNG, JPEG, WebP, GIF)
- Format conversion utilities
- Provider status management
- Generation result handling
- Provider model configuration
- Base provider interface compliance

### Pollinations Provider Tests (`test_pollinations.py`)
- Successful image generation
- Model selection and configuration
- Dimension clamping
- HTTP error handling
- Timeout handling
- Network error handling
- URL encoding
- Health checks
- Request tracking

### Server Tools Tests (`test_server.py`)
- `generate_image` tool with all parameters
- `generate_pixel_art` tool with style variants
- `list_providers` tool
- `list_models` tool
- `get_provider_status` tool
- `save_image` tool
- Provider fallback mechanism
- File saving functionality

### Validation Tests (`test_validation.py`)
- Invalid dimension handling
- Special characters in prompts
- Very long prompts
- Invalid provider/model names
- Invalid seed values
- Pixel art parameter validation
- Invalid base64 data
- Invalid filenames
- Unicode support
- Concurrent requests
- Null value handling
- Error recovery

## Running Tests

### Run all tests:
```bash
pytest
```

### Run with coverage:
```bash
pytest --cov=src/image_gen_mcp --cov-report=html
```

### Run specific test file:
```bash
pytest tests/test_base.py
pytest tests/test_pollinations.py
pytest tests/test_server.py
pytest tests/test_validation.py
```

### Run specific test class:
```bash
pytest tests/test_base.py::TestImageFormat
pytest tests/test_pollinations.py::TestPollinationsProvider
```

### Run specific test:
```bash
pytest tests/test_base.py::TestImageFormat::test_image_format_extensions
```

### Run tests by marker:
```bash
pytest -m asyncio          # Only async tests
pytest -m "not slow"       # Skip slow tests
pytest -m unit             # Only unit tests
```

### Run with verbose output:
```bash
pytest -v
pytest -vv                 # Extra verbose
```

### Run with output capture disabled (see print statements):
```bash
pytest -s
```

## Test Fixtures

### Available in `conftest.py`:

- `sample_png_bytes` - Valid 1x1 PNG image bytes
- `sample_jpeg_bytes` - Valid 1x1 JPEG image bytes
- `sample_png_base64` - Base64-encoded PNG
- `sample_jpeg_base64` - Base64-encoded JPEG
- `mock_output_dir` - Temporary output directory
- `mock_provider` - Mock image provider for testing
- `mock_aiohttp_response` - Mock HTTP response
- `mock_aiohttp_session` - Mock HTTP session

## Coverage Goals

- **Overall Coverage**: >85%
- **Base Module**: >90%
- **Server Module**: >85%
- **Provider Modules**: >80%

## Current Coverage

Run `pytest --cov` to see current coverage statistics.

View detailed HTML coverage report:
```bash
pytest --cov-report=html
open htmlcov/index.html
```

## Writing New Tests

### Test Naming Convention:
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test:
```python
import pytest
from image_gen_mcp.providers.base import ImageFormat

class TestMyFeature:
    """Test description."""

    def test_basic_functionality(self):
        """Test basic case."""
        result = ImageFormat.PNG
        assert result.extension == ".png"

    @pytest.mark.asyncio
    async def test_async_functionality(self, mock_provider):
        """Test async case."""
        result = await mock_provider.generate(prompt="test")
        assert result.success is True
```

## Continuous Integration

Tests run automatically on:
- Push to main branch
- Pull requests
- Pre-commit hooks (if configured)

## Troubleshooting

### Import Errors
Ensure the package is installed in development mode:
```bash
pip install -e .
```

### Async Test Errors
Make sure `pytest-asyncio` is installed:
```bash
pip install pytest-asyncio
```

### Coverage Not Found
Install coverage tools:
```bash
pip install pytest-cov
```

### Tests Timing Out
Increase timeout in `pytest.ini` or use `-o timeout=30` flag.

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure >80% coverage for new code
3. Run full test suite before committing
4. Update this README if adding new test files
