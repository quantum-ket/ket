from setuptools import setup, find_namespace_packages
from ket.__version__ import __version__

with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name            = 'ket-lang',
    description     = 'Ket Quantum Programming Language interpreter and library.',
    url             = 'https://quantumket.org',
    version         = __version__,
    author          = 'Evandro Chagas Ribeiro da Rosa',
    author_email    = 'ev.crr97@gmail.com',
    license         = 'Apache-2.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    packages=find_namespace_packages(include=['ket', 'ket.*']),
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
    ],
    entry_points={'console_scripts': ['ket = ket.__main__:__ket__']},
)
