import logging
import os
from datetime import datetime
from typing import Optional


class ETLLogger:
    """Centralized logger handler for ETL pipeline using Singleton pattern."""

    LOG_DIR = "logs"
    LOG_LEVEL = logging.INFO
    MAX_LOG_FILES = 2  # Keep only 2 most recent log files

    _instance = None
    _initialized = False

    def __new__(
        cls,
        name: str = "ETL_Pipeline",
        log_dir: Optional[str] = None,
        console_output: bool = True,
    ):
        """Implement singleton pattern - return same instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name: str = "ETL_Pipeline",
        log_dir: Optional[str] = None,
        console_output: bool = True,
    ):
        """
        Initialize logger instance (only once due to singleton).

        Args:
            name: Logger name.
            log_dir: Directory to save logs (default: logs/).
            console_output: Whether to also output to console/stdout (default: True).
        """
        # Only initialize once
        if ETLLogger._initialized:
            return

        self.name = name
        self.log_dir = log_dir or self.LOG_DIR
        self.console_output = console_output
        os.makedirs(self.log_dir, exist_ok=True)

        # Clean up old logs before creating new one
        self._cleanup_old_logs()

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LOG_LEVEL)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler - logs to file
        self._setup_file_handler(formatter)

        # Console handler - logs to stdout if enabled
        if self.console_output:
            self._setup_console_handler(formatter)

        ETLLogger._initialized = True

    def _cleanup_old_logs(self) -> None:
        """Delete old log files, keeping only the MAX_LOG_FILES most recent."""
        try:
            log_files = []

            # Find all log files in the log directory
            for filename in os.listdir(self.log_dir):
                if filename.startswith("etl_pipeline_") and filename.endswith(".log"):
                    filepath = os.path.join(self.log_dir, filename)
                    # Get file modification time
                    mod_time = os.path.getmtime(filepath)
                    log_files.append((filepath, mod_time))

            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: x[1], reverse=True)

            # Delete files older than MAX_LOG_FILES
            if len(log_files) > self.MAX_LOG_FILES:
                for filepath, _ in log_files[self.MAX_LOG_FILES :]:
                    try:
                        os.remove(filepath)
                        print(f"Deleted old log file: {filepath}")
                    except Exception as e:
                        print(f"Failed to delete log file {filepath}: {e}")
        except Exception as e:
            print(f"Error during log cleanup: {e}")

    def _setup_file_handler(self, formatter: logging.Formatter) -> None:
        """Setup file handler for logging to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"etl_pipeline_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.LOG_LEVEL)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.log_file = log_file

    def _setup_console_handler(self, formatter: logging.Formatter) -> None:
        """Setup console handler for logging to stdout."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.LOG_LEVEL)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    # ==================== SINGLETON UTILITY METHODS ====================

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)."""
        if cls._instance and hasattr(cls._instance, "close"):
            cls._instance.close()
        cls._instance = None
        cls._initialized = False

    # ==================== LOGGING METHODS ====================

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    def exception(self, message: str) -> None:
        """Log exception with traceback."""
        self.logger.exception(message)

    def get_log_file(self) -> str:
        """Get path to current log file."""
        return self.log_file

    def close(self) -> None:
        """Close all handlers."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
