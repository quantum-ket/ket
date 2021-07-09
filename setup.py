import subprocess
import setuptools
import sys

try:
    from skbuild import setup
except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'scikit-build'])
    from skbuild import setup

with open('README.md', 'r') as file:
    long_description = file.read()

setup_requirements = ['scikit-build>=0.11.1',
                      'cmake>=3.15',
                      'conan>=1.25'
                      ]

setup (
    name = 'ket-lang',
    version='0.1.dev0',
    cmake_source_dir='.',
    cmake_args=['-DCMAKE_BUILD_TYPE=Release'],
    author='Evandro Chagas Ribeiro da Rosa',
    author_email='evandro.crr@posgrad.ufsc.br',
    description='Ket Quantum Programming Language interpreter and library.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/quantum-ket/ket',
    license='Apache-2.0',
    packages=setuptools.find_namespace_packages(include=['ket', 'ket.*']),
    setup_requires=setup_requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    entry_points={'console_scripts': ['ket = ket.__main__:__ket__']},
)
