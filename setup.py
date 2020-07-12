from skbuild import setup
from setuptools import find_namespace_packages
from os import listdir

with open('README.md', 'r') as file:
    long_description = file.read()

setup_requirements = ['scikit-build>=0.11.1',
                      'cmake>=3.15',
                      'conan>=1.25'
                      ]

setup (name = 'ket',
       version='0.1b',
       cmake_source_dir='.',
       author='Evandro Chagas Ribeiro da Rosa',
       author_email='evandro.crr@posgrad.ufsc.br',
       description='Ket quantum programming language library and interpreter',
       long_description=long_description,
       long_description_content_type='text/markdown',
       url='https://gitlab.com/quantum-ket/ket',
       packages=find_namespace_packages(),
       setup_requires=setup_requirements,
       classifiers=['Programming Language :: Python :: 3',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: MIT License',
                   ]
       )
