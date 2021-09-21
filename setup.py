import subprocess
import setuptools
import sys
from datetime import datetime

def git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode()[:-1] + ", "
    except:
        return ""

try:
    from skbuild import setup
except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'scikit-build'])
    from skbuild import setup

with open('README.md', 'r') as file:
    long_description = file.read()

setup_requirements = [
    'scikit-build>=0.11',
    'cmake>=3.15', 
    'ninja>=1.10', 
    'conan>=1.25'
]

__version__ = '0.1.1'

setup(
    name            = 'ket-lang',
    description     = 'Ket Quantum Programming Language interpreter and library.',
    url             = 'https://quantum-ket.gitlab.io',
    version         = __version__,
    author          = 'Evandro Chagas Ribeiro da Rosa',
    author_email    = 'evandro.crr@posgrad.ufsc.br',
    license         = 'Apache-2.0',
    cmake_source_dir='.',
    cmake_args=[
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DKET_BUILD_INFO="{__version__} ({git_hash()}{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")})"',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
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
