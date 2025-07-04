"""
Helper functions for interacting with Weights & Biases (W&B) artifacts.
"""

import wandb


def artifact_file_exists(project_name, artifact_name, file_path):
    """
    Check if a specific file exists within a W&B artifact.

    Parameters:
    - project_name: str, e.g. "username/project"
    - artifact_name: str, e.g. "artifact-name:version" or "artifact-name:latest"
    - file_path: str, path within the artifact (e.g., "branch_weights.h5")

    Returns:
    - bool: True if the file exists in the artifact, False otherwise
    """
    api = wandb.Api()
    try:
        artifact = api.artifact(f"{project_name}/{artifact_name}", type="model")
        return file_path in [f.path for f in artifact.manifest.entries.values()]
    except wandb.errors.CommError:
        return False
