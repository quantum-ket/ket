# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""C Libs"""

import os
import shutil
import warnings


def libs():
    """Name Rust libs"""
    if os.name == "nt":
        return [("libket", "ket.dll"), ("kbw", "kbw.dll")]
    if os.uname().sysname == "Linux":
        return [("libket", "libket.so"), ("kbw", "libkbw.so")]
    if os.uname().sysname == "Darwin":
        return [("libket", "libket.dylib"), ("kbw", "libkbw.dylib")]
    raise OSError("unsupported operational system")


def make_libs():
    """Make Rust libs if missing"""
    dirname = os.path.dirname(__file__)

    if all(os.path.isfile(f"{dirname}/{lib}") for _, lib in libs()):
        return

    warnings.warn(
        "Compiling Libket and KBW... may take a while. \
If it fails, check if Rust is installed https://www.rust-lang.org/tools/install."
    )

    os.chdir(f"{dirname}/libket")
    os.system("cargo build --release --quiet")
    os.chdir("../kbw")
    os.system("cargo build --release --quiet")

    for dirlib, lib in libs():
        src = f"{dirname}/{dirlib}/target/release/{lib}"
        dst = f"{dirname}/{lib}"
        shutil.move(src, dst)


make_libs()
