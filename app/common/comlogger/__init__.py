import logging
import logging.handlers
import os
import ecs_logging

from ..comconfig import settings
from ..comconstants import DefaultKeys, DefaultValues


# Get the Logger
def get_shared_logger(
    to_file=True, log_path=f"{os.getcwd()}/logs/app-{os.getpid()}.log"
):
    logger = logging.getLogger(
        settings.get(DefaultKeys.APP__NAME, DefaultValues.APP_DEFAULT_NAME)
    )
    app_env = settings.get(DefaultKeys.APP__ENV)
    if app_env == DefaultValues.APP_ENV_DEBUG:
        logger.setLevel(logging.DEBUG)
    elif app_env == DefaultValues.APP_ENV_PROD:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.INFO)

    # Add an ECS formatter to the Handler
    handler = logging.StreamHandler()
    handler.setFormatter(ecs_logging.StdlibFormatter())
    if not logger.hasHandlers():
        logger.addHandler(handler)
        if to_file:
            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_path, interval=10, when="D"
            )
            file_handler.setFormatter(ecs_logging.StdlibFormatter())
            logger.addHandler(file_handler)
    logger.propagate = False

    return logger
