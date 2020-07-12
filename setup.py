from setuptools import setup
from setuptools.extension import Extension
from os import listdir

with open('README.md', 'r') as file:
    long_description = file.read()

libket_src = ['libket/src/'+file for file in listdir('libket/src')]
libket_src.append('build/ket/ketPYTHON_wrap.cxx')

ext_module = Extension('ket._ketpy',
        sources=libket_src, 
        extra_compile_args=['-std=c++17'],
        )

setup (name = 'ket',
       version='0.1rc1',
       author='Evandro Chagas Ribeiro da Rosa',
       author_email='evandro.crr@posgrad.ufsc.br',
       description='Ket quantum programming language library and interpreter',
       long_description=long_description,
       long_description_content_type='text/markdown',
       url='https://gitlab.com/quantum-ket/ket',
       ext_modules = [ext_module],
       packages=['build/ket'],
       classifiers=[
            'Programming Language :: Python :: 3',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
        ]
       )
