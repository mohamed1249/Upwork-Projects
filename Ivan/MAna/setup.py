from setuptools import setup, find_packages

setup(
  name='MAna',
  version='0.0.1',
  description='A Python Package for Data Manipulation and Analysis',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='M_A', 
  keywords='data', 
  packages=find_packages(),
  install_requires=[''] 
)