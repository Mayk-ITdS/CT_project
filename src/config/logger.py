import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        handlers=[
            logging.FileHandler("runtime.log"),
            logging.StreamHandler()
        ]
    )