import logging
import json
import sys

class JSONLFormatter(logging.Formatter):
    def format(self, record):
        log_record: dict
        # If the message is a dict, use it
        if isinstance(record.msg, dict):
            log_record = record.msg.copy()
        # Otherwise convert it to a dict
        else:
            log_record = {"message": record.getMessage()}

        # Add required log metadata
        log_record.update({
            "logger": record.name,
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
        })

        return json.dumps(log_record)
    

class JSONLLogger:
    def __init__(self, name = None, level=logging.INFO, log_file=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        handler_console = logging.StreamHandler()
        handler_console.setFormatter(JSONLFormatter())
        self.logger.addHandler(handler_console)

        if log_file:
            handler_file = logging.FileHandler(log_file)
            handler_file.setFormatter(JSONLFormatter())
            self.logger.addHandler(handler_file)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


# This will log to both the console and the 'app.log' file
logger = JSONLLogger("MyApp", level=logging.DEBUG, log_file="app.log")
logger.debug("This is a debug message.")
logger.info({"message": "This is an info message with some extra stuff", "key1": "value1", "key2": 2, "key3": True})
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")

# This will only log to the console
console_only_logger = JSONLLogger("ConsoleLogger")
console_only_logger.info("This message only goes to the console.")