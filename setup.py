from setuptools import setup , find_packages
from typing import List

def get_requirement(path:str)->list[str]:
    requirement=[]
    with open(path) as p:
        requirement = p.readlines()
        requirement=[req.replace('\n' , '') for req in requirement]
        print(requirement)    
    
    if '-e .' in requirement:
        requirements=requirement.remove('-e .')
        
    print(requirements)
         # return requirement


setup(
    name='Machine Learning Project' , 
    version='0.0.3' , 
    author='Pattani Princi' , 
    author_email='pattaniprinci@gmail.com' , 
    packages=find_packages() , 
    install_requires=get_requirement('requirement.txt')
)

