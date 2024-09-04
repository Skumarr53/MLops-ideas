import logging
import uuid

class RequestContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = uuid.uuid4().hex
        return True

def setup_logging(log_file_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s',
        filename=log_file_path,
        filemode='w'
    )
    logging.getLogger().addFilter(RequestContextFilter())