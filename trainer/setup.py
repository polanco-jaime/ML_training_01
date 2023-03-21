from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs==0.7.1', 
    'dask[dataframe]==2021.2.0', 
    'google-cloud-bigquery-storage==1.0.0', 
    'six==1.15.0', 
    'lightgbm==3.3.5'
]
 
setup(
    name='trainer', 
    version='0.1', 
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(), # Encuentra automaticamente paquetes dentro de este directorio.
    include_package_data=True, # Si los paquetes incluyen archivos de datos, se empaquetan juntos.
    description='da una descripcion'
)
