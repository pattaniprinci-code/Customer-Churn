import os
import sys
from src.ML_PROJECT.components import logger
from src.ML_PROJECT.exception import customException
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text


def read_sql_data():
    logger.logging.info("start reading data ")
    try:
        engine = create_engine("postgresql://postgres:1234@localhost:5432/customer_churn")
        logger.logging.info("my sql connection successfully")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM customer_churn_data"))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        print(df.head())
        print('Data read successfully')
        logger.logging.info(f"sample data :- {df.head}")
        return df

    except Exception as e:
        logger.logging.info(customException(e, sys))
        raise customException(e, sys)
