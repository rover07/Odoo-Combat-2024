from setuptools import find_packages,setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of all the requirements mentioned in the file
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements] # remove new line characters

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name= "Odoo-Hackathon",
    version= "0.0.1",
    author= "Team",
    author_email= "",
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)