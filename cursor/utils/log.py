import logging


def get_logger(
    name: str,
    level=logging.DEBUG,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    return logger
