import os
import sys
import unittest
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_models_config, load_server_config


class TestConfig(unittest.TestCase):
    """Test configuration loading."""
    
    def test_load_models_config(self):
        """Test loading models configuration."""
        models_config = load_models_config("config/models_config.yaml")
        self.assertIsNotNone(models_config)
        self.assertGreater(len(models_config.models), 0)
        
        # Check that each model has required fields
        for model in models_config.models:
            self.assertIsNotNone(model.id)
            self.assertIsNotNone(model.name)
            self.assertIsNotNone(model.huggingface_repo)
            self.assertIsNotNone(model.type)
            self.assertIsNotNone(model.family)
            self.assertGreater(model.context_length, 0)
            self.assertGreater(model.ram_required, 0)
            self.assertGreater(model.gpu_required, 0)
            self.assertGreater(len(model.quantization), 0)
            self.assertGreater(len(model.tags), 0)
            self.assertIsNotNone(model.description)
    
    def test_load_server_config(self):
        """Test loading server configuration."""
        server_config = load_server_config("config/server_config.yaml")
        self.assertIsNotNone(server_config)
        
        # Check server section
        self.assertIsNotNone(server_config.server.host)
        self.assertIsNotNone(server_config.server.port)
        self.assertIsNotNone(server_config.server.api_prefix)
        
        # Check models section
        self.assertIsNotNone(server_config.models.cache_dir)
        self.assertIsNotNone(server_config.models.max_loaded_models)
        
        # Check inference section
        self.assertIsNotNone(server_config.inference.max_length)
        self.assertIsNotNone(server_config.inference.temperature)
        
        # Check security section
        self.assertIsNotNone(server_config.security.enable_auth)
        self.assertIsNotNone(server_config.security.allowed_origins)
        
        # Check logging section
        self.assertIsNotNone(server_config.logging.level)
        self.assertIsNotNone(server_config.logging.file)


if __name__ == "__main__":
    unittest.main()