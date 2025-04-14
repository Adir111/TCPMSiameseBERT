import wandb

_logger_instance = None


class NoOpWandb:
    _instance = None

    def __init__(self):
        self.config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NoOpWandb, cls).__new__(cls)
            cls._instance.config = {}
        return cls._instance

    def init(self, project=None, name=None, config=None):
        self.config = config or {}
        print(f"[NoOpWandb] Initialized with project='{project}', name='{name}'")

    @staticmethod
    def log(data):
        print(f"[NoOpWandb] Logging: {data}")

    @staticmethod
    def watch(model=None, **kwargs):
        print(f"[NoOpWandb] Watching model: {model}")

    @staticmethod
    def save(path):
        print(f"[NoOpWandb] Saving file: {path}")

    @staticmethod
    def finish():
        print("[NoOpWandb] Finished run")

    @staticmethod
    def Histogram(*args, **kwargs):
        print(f"[NoOpWandb] Creating histogram with args={args}, kwargs={kwargs}")
        return None

    @staticmethod
    def log_summary(summary_name, name):
        print(f"[NoOpWandb] Logging summary with name={name}")

    def __getattr__(self, name):
        def no_op(*args, **kwargs):
            print(f"[NoOpWandb] Called undefined method: {name} with args={args}, kwargs={kwargs}")

        return no_op


class WrappedWandbLogger:
    _instance = None

    def __new__(cls, wandb_instance):
        if cls._instance is None:
            cls._instance = super(WrappedWandbLogger, cls).__new__(cls)
            cls._instance._wandb = wandb_instance
        return cls._instance

    def init(self, **kwargs):
        return self._wandb.init(**kwargs)

    def save(self, *args, **kwargs):
        return self._wandb.save(*args, **kwargs)

    def watch(self, *args, **kwargs):
        return self._wandb.watch(*args, **kwargs)

    def finish(self):
        return self._wandb.finish()

    def Histogram(self, *args, **kwargs):
        return self._wandb.Histogram(*args, **kwargs)

    def log_summary(self, summary_name, name):
        self._wandb.run.summary[summary_name] = name

    def log(self, data):
        if isinstance(data, str):
            if hasattr(self._wandb, "termlog"):
                self._wandb.termlog(data)
            else:
                print(f"[NoOpWandb] {data}")
        elif isinstance(data, dict):
            self._wandb.log(data)
        else:
            raise TypeError(f"log() accepts either a string or a dictionary. Got: {type(data)}")


def get_logger(config):
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    if config.get("wandb", {}).get("enabled", False):
        wandb.login(key=config["wandb"]["api_key"])
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config,
        )
        _logger_instance = WrappedWandbLogger(wandb)
        _logger_instance.log("[INFO] WandB logger initialized")
    else:
        _logger_instance = NoOpWandb()
        _logger_instance.log("[INFO] NoOpWandb logger initialized")

    return _logger_instance

