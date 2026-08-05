from src.ML_PROJECT.components import logger
from src.ML_PROJECT.exception import customException
import os
import sys
from src.ML_PROJECT.components.data_ingestion import DataIngestion

try:
    DataIngestionobj = DataIngestion()
    logger.logging.info("function startrd")
    DataIngestionobj.read_data()
    logger.logging.info("function call")

except Exception as e:
    logger.logging.info(customException(e, sys))
    raise Exception(e, sys)
