import logging


def get_logger(
    name: str,
    level=logging.DEBUG,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handler already exists to prevent duplicates
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False
    
    return logger
