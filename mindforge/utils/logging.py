import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class LogManager:
    """Manage application logging."""

    def __init__(
        self, log_dir: str = "logs", log_level: Union[str, int] = logging.INFO
    ):
        self.log_dir = Path(log_dir)
        self.log_level = (
            log_level.upper() if isinstance(log_level, str) else log_level
        )  # Handle string levels
        self.loggers = {}

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup default logger
        self.setup_logger("mindforge")

    def setup_logger(
        self, name: str, log_file: Optional[str] = None
    ) -> logging.Logger:
        """Setup a new logger."""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        # Create formatters
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Create file handler if log_file specified
        if log_file:
            file_path = self.log_dir / log_file
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        self.loggers[name] = logger
        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """Get an existing logger or create a new one."""
        if name not in self.loggers:
            # Use a consistent log file naming convention
            return self.setup_logger(name, f"{name}.log")
        return self.loggers[name]