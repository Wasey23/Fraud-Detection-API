import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a custom logger with a standardized,
    professional formatting structure for terminal output.
    """
    # Initialize the logger with the name of the module calling it
    logger = logging.getLogger(name)

    # Prevent duplicate logs if the logger is called multiple times
    if not logger.handlers:
        # Set the threshold for logging (INFO and above will be displayed)
        logger.setLevel(logging.INFO)

        # Create a handler that outputs to the terminal (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Define a clean, time-stamped format structure
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Attach the formatter to the handler, and the handler to the logger
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger