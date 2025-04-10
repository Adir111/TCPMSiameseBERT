import os
import yaml


class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
        with open(config_path, "r") as f:
            cls._config = yaml.safe_load(f)

    def get_config(self):
        return self._config


def get_config():
    return ConfigLoader().get_config()
