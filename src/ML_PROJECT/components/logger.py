from datetime import datetime
import os
import logging


LOG_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_dir, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

LOG_PATH = os.path.join(LOG_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
 format="[%(asctime)s] %(levelname)-8s | %(name)s | Line:%(lineno)d | %(funcName)s() | %(message)s",
)
