"""
Provides a singleton ConfigLoader class to load and access application configuration
from a YAML file. Ensures configuration is loaded once and reused throughout the application.

Includes a helper function get_config() to easily retrieve the loaded configuration.
"""

import os
import yaml


class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        """
        Create a singleton instance of ConfigLoader.
        Loads configuration from YAML file on first instantiation.
        """
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        """
        Load the configuration from the YAML file into the class variable _config.
        """
        config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
        with open(config_path, "r") as f:
            cls._config = yaml.safe_load(f)

    def get_config(self):
        """
        Return the loaded configuration dictionary.
        """
        return self._config


def get_config():
    """
    Helper function to get the singleton configuration instance's config dictionary.
    """
    return ConfigLoader().get_config()
