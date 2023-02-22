#  Copyright 2020, 2023 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import shutil


def libs():
    """Name Rust libs"""
    if os.name == 'nt':
        return [('libket', 'ket.dll'), ('kbw', 'kbw.dll')]
    if os.uname().sysname == 'Linux':
        return [('libket', 'libket.so'), ('kbw', 'libkbw.so')]
    if os.uname().sysname == 'Darwin':
        return [('libket', 'libket.dylib'), ('kbw', 'libkbw.dylib')]
    raise OSError('unsupported operational system')


def make_libs():
    """Make Rust libs if missing"""
    dirname = os.path.dirname(__file__)

    if all(os.path.isfile(f"{dirname}/{lib}") for _, lib in libs()):
        return

    os.chdir(f"{dirname}/libket")
    os.system("cargo build --release --quiet")
    os.chdir("../kbw")
    os.system("cargo build --release --quiet")

    for dirlib, lib in libs():
        src = f"{dirname}/{dirlib}/target/release/{lib}"
        dst = f"{dirname}/{lib}"
        shutil.move(src, dst)


make_libs()
