from setuptools import setup, find_namespace_packages
import os
import shutil


def libs():
    if os.name == 'nt':
        return [('libket', 'ket.dll'), ('kbw', 'kbw.dll')]
    else:
        return [('libket', 'libket.so'), ('kbw', 'libkbw.so')]


def make_libs():
    try:
        os.mkdir("libs")
    except FileExistsError:
        pass
    os.chdir("libs")
    os.system(
        "git clone https://gitlab.com/quantum-ket/libket.git || git -C libket pull")
    os.system("git clone https://gitlab.com/quantum-ket/kbw.git || git -C kbw pull")

    os.chdir("libket")
    os.system("cargo build --release")
    os.chdir("../kbw")
    os.system("cargo build --release")

    os.chdir("../..")
    try:
        os.mkdir("ket/clib/libs")
    except FileExistsError:
        pass

    print(os.getcwd())
    for dir, lib in libs():
        src = f"libs/{dir}/target/release/{lib}"
        dst = "ket/clib/libs/"+lib
        shutil.copy(src, dst)


make_libs()


def long_description():
    with open('README.md', 'r') as file:
        return file.read()


def version():
    from ket.__version__ import __version__
    return __version__


setup(
    name='ket-lang',
    description='Ket Quantum Programming Language interpreter and library.',
    url='https://quantumket.org',
    version=version(),
    author='Evandro Chagas Ribeiro da Rosa',
    author_email='ev.crr97@gmail.com',
    license='Apache-2.0',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    include_package_data=True,
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
