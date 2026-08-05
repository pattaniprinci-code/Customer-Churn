import os 
from pathlib import Path 
import logging

logging.basicConfig(level=logging.INFO)
project_name='ML_PROJECT'

list_of_files=[
    f'src/{project_name}/__init__.py' ,
    f'src/{project_name}/components/__init__.py' ,
    f'src/{project_name}/components/data_ingestion.py' ,
    f'src/{project_name}/components/data_transformation.py' ,
    f'src/{project_name}/components/data_trainer.py' ,
    f'src/{project_name}/components/data_monitoring.py' ,
    f'src/{project_name}/components/__init__.py' ,
    f'src/{project_name}/components/training_pipeline.py' ,
    f'src/{project_name}/components/prediction_pipeline.py' ,
    'app.py' ,
    'Dockerfile' ,
    'requirements.txt' ,
    'setup.py' , 
    'main.py', 
    
]

for file_path in list_of_files:
    file_path=Path(file_path)
    file_dir, file_name=os.path.split(file_path) 
    print('file dir' , file_dir)
    print('file name' , file_name)
    
    if file_dir !='':
        os.makedirs(file_dir , exist_ok=True)
        
    if not os.path.exists(file_path) or (os.path.getsize(file_path) == 0):
        with open (file_path , 'w') as f:
            pass
        