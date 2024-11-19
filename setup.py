from setuptools import setup, find_packages
from typing import List

HYPEN_DOT = "-e ."


def get_install_packages(file_name: str) -> List[str]:
    requirements = []
    with open(file_name) as file_object:
        requirements = file_object.readlines()
        requirements = [
            req.replace("\n", "") for req in requirements if req != HYPEN_DOT
        ]

    return requirements


setup(
    name="Material_Lifespan_Prediction",
    version="0.0.1",
    description="Package to predict the material lifespan based on the material various features.",
    author="Shivakumar Ravichandran",
    author_email="shivakumar.mcet@gmail.com",
    packages=find_packages(),
    install_requires=get_install_packages("requirements.txt"),
)
