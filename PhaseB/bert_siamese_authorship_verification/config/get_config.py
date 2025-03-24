import os
import yaml


def get_config():
    # Determine correct path for config.yaml
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "config/config.yaml")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
