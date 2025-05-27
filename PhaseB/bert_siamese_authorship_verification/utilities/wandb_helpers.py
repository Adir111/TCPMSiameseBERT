import wandb


def artifact_file_exists(project_name, artifact_name, file_path):
    """
    Check if a specific file exists within a W&B artifact.

    :param project_name: e.g. "username/project"
    :param artifact_name: e.g. "artifact-name:version" or "artifact-name:latest"
    :param file_path: path within the artifact (e.g., "branch_weights.h5")
    :return: True if the file exists, else False
    """
    api = wandb.Api()
    try:
        artifact = api.artifact(f"{project_name}/{artifact_name}", type="model")
        return file_path in [f.path for f in artifact.manifest.entries.values()]
    except wandb.errors.CommError:
        return False
