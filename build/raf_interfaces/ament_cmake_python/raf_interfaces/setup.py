from setuptools import find_packages
from setuptools import setup

setup(
    name='raf_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('raf_interfaces', 'raf_interfaces.*')),
)
