import os
import yaml


def get_config():
    # Determine correct path for config.yaml
    config_path = os.path.join(os.path.dirname(__file__), "./config.yaml")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
