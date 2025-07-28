def get_logger():
    from loguru import logger
    # Add a file sink to write logs to a file
    logger.add(
        "logs/app.log",
        rotation="5 MB",
        retention="7 days",
        level="INFO"
        )
    return logger

# initiallize the logger
logger = get_logger()