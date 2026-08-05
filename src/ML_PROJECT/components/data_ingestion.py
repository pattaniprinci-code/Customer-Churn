import os 
import sys 
import pandas as pd 
from src.ML_PROJECT.components import logger 
from src.ML_PROJECT.exception import customException
from dataclasses import dataclass
from src.ML_PROJECT.utils import read_sql_data
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join("Artifacts" , "train.csv")
    test_data_path : str=os.path.join('Artifacts' , 'test.csv')
    raw_data_path : str= os.path.join ('Artifacts' , 'raw . csv')
    
class DataIngestion:
    def __init__(self):
        self.ingetionconfig=DataIngestionConfig()
        
    def read_data(self):
        try:
            logger.logging.info('start Reading Data From Sql')
            df=read_sql_data()
            os.makedirs(os.path.dirname(self.data_ingestion.row_data_path) , exist_ok=True)
            
            df.to_csv(self.data_ingetion.row_data_path , index=False , header=True)
            logger.logging.info('raw.csv created successfully')

            train_data_set , test_data_set = train_test_split(
                df , test_size=0.20 , random_state=42
            )
            train_data_set.to_csv(
                self.data_ingestion.test_data_path , index=False , header=True
                
            )
            logger.logging.info('train.csv created successfully...')
            test_data_set.to_csv(
                self.data_ingestion.test_data_path , index=False , header=True
             )
            logger.logging.info('test.csv file created successfully......')
            return(
                self.data_ingestion.train_data_path , 
                self.data_ingestion.test_data_path , 
                
            )
            
            
            
        except Exception as e :
            raise customException(e , sys)        
    

            