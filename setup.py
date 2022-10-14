from setuptools import setup
import os
import shutil


def libs():
    if os.name == 'nt':
        return [('libket', 'ket.dll'), ('kbw', 'kbw.dll')]
    if os.uname().sysname == 'Linux':
        return [('libket', 'libket.so'), ('kbw', 'libkbw.so')]
    if os.uname().sysname == 'Darwin':
        return [('libket', 'libket.dylib'), ('kbw', 'libkbw.dylib')]
    raise OSError('unsupported operational system')


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
        os.mkdir("src/ket/clib/libs")
    except FileExistsError:
        pass

    for dir, lib in libs():
        src = f"libs/{dir}/target/release/{lib}"
        dst = "src/ket/clib/libs/"+lib
        shutil.copy(src, dst)


make_libs()

setup()
