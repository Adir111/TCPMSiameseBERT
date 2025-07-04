"""
Logger module providing a WandB wrapper and a no-op logger fallback.

Defines:
- NoOpWandb: A dummy logger class for disabled WandB scenarios.
- WrappedWandbLogger: A singleton wrapper around the real WandB instance.
- get_logger: Factory to return a configured logger instance based on config.
"""

import wandb

_logger_instance = None


class NoOpWandb:
    """
    No-operation WandB logger stub.

    Mimics the WandB API but outputs logs to the console.
    Implements singleton pattern.
    """
    _instance = None

    def __init__(self):
        self.config = None

    def __new__(cls):
        """
        Ensures a single instance of the NoOpWandb class (singleton pattern).

        Returns:
            NoOpWandb: Singleton instance of the NoOpWandb logger.
        """
        if cls._instance is None:
            cls._instance = super(NoOpWandb, cls).__new__(cls)
            cls._instance.config = {}
        return cls._instance

    def init(self, project=None, name=None, config=None):
        """
        Initialize the no-op logger.
        """
        self.config = config or {}
        print(f"[NoOpWandb] Initialized with project='{project}', name='{name}'")

    @staticmethod
    def info(data):
        """Log info message."""
        print(f"[NoOpWandb] Info: {data}")

    @staticmethod
    def warn(data):
        """Log warning message."""
        print(f"[NoOpWandb] Warning: {data}")

    @staticmethod
    def error(data):
        """Log error message."""
        print(f"[NoOpWandb] Error: {data}")

    @staticmethod
    def log(data):
        """Log general data."""
        print(f"[NoOpWandb] {data}")

    @staticmethod
    def watch(model=None, **kwargs):
        """Stub for WandB watch."""
        print(f"[NoOpWandb] Watching model: {model}")

    @staticmethod
    def save(path):
        """Stub for WandB save."""
        print(f"[NoOpWandb] Saving file: {path}")

    @staticmethod
    def finish():
        """Stub for WandB finish."""
        print("[NoOpWandb] Finished run")

    @staticmethod
    def Histogram(*args, **kwargs):
        """Stub for WandB Histogram."""
        print(f"[NoOpWandb] Creating histogram with args={args}, kwargs={kwargs}")
        return None

    @staticmethod
    def log_summary(summary_name, name):
        """Stub for logging summary."""
        print(f"[NoOpWandb] Logging summary with name={name}")

    def __getattr__(self, name):
        """
        Catch-all for undefined method calls, printing debug info.
        """
        def no_op(*args, **kwargs):
            print(f"[NoOpWandb] Called undefined method: {name} with args={args}, kwargs={kwargs}")

        return no_op


class WrappedWandbLogger:
    """
    Singleton wrapper for the real WandB logger instance.

    Provides convenient logging methods, forwarding to WandB.
    """

    _instance = None

    def __new__(cls, wandb_instance):
        """
        Ensures a single instance of the WrappedWandbLogger class (singleton pattern).

        Args:
            wandb_instance: The actual WandB module or object to wrap.

        Returns:
            WrappedWandbLogger: Singleton instance wrapping the provided WandB object.
        """
        if cls._instance is None:
            cls._instance = super(WrappedWandbLogger, cls).__new__(cls)
            cls._instance.wandb = wandb_instance
        return cls._instance

    def init(self, **kwargs):
        """Initialize WandB run."""
        return self.wandb.init(**kwargs)

    def save(self, *args, **kwargs):
        """Save files in WandB."""
        return self.wandb.save(*args, **kwargs)

    def watch(self, *args, **kwargs):
        """Watch model in WandB."""
        return self.wandb.watch(*args, **kwargs)

    def finish(self):
        """Finish WandB run."""
        return self.wandb.finish()

    def Histogram(self, *args, **kwargs):
        """Create WandB histogram."""
        return self.wandb.Histogram(*args, **kwargs)

    def Image(self, *args, **kwargs):
        """Create WandB image."""
        return self.wandb.Image(*args, **kwargs)

    def log_summary(self, summary_name, name):
        """Log summary data in WandB run."""
        self.wandb.run.summary[summary_name] = name

    def info(self, data):
        """Log info level message."""
        self.log(data, '[INFO] ')

    def warn(self, data):
        """Log warning level message."""
        self.log(data, '[WARNING] ')

    def error(self, data):
        """Log error level message."""
        self.log(data, '[ERROR] ')

    def log(self, data, prefix=''):
        """
        Log a message or dictionary to WandB.

        Accepts strings or dicts. Strings are logged with optional prefix.
        """
        if isinstance(data, str):
            if hasattr(self.wandb, "termlog"):
                self.wandb.termlog(prefix + data)
            else:
                raise AttributeError("WandB instance does not have 'termlog' method.")
        elif isinstance(data, dict):
            self.wandb.log(data)
        else:
            raise TypeError(f"log() accepts either a string or a dictionary. Got: {type(data)}")


def get_logger(config):
    """
    Get or create a singleton logger instance based on configuration.

    If WandB is enabled in config, returns a WrappedWandbLogger initialized with WandB.
    Otherwise, returns a NoOpWandb instance.

    Parameters:
    - config: dictionary containing logger configuration

    Returns:
    - Logger instance with WandB-compatible interface
    """
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
        _logger_instance.info("WandB logger initialized")
    else:
        _logger_instance = NoOpWandb()
        _logger_instance.info("NoOpWandb logger initialized")

    return _logger_instance

