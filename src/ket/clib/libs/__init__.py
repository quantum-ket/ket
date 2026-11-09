# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""C Libs"""

import os
import platform
import shutil
import subprocess
import warnings

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# KBW features and the environment variables that enable or disable them.
_KBW_FEATURES = (("gpu", "KET_BUILD_GPU"), ("cuda", "KET_BUILD_CUDA"))


def libs():
    """Name Rust libs"""
    if os.name == "nt":
        return [("libket", "ket.dll"), ("kbw", "kbw.dll")]
    if os.uname().sysname == "Linux":
        return [("libket", "libket.so"), ("kbw", "libkbw.so")]
    if os.uname().sysname == "Darwin":
        return [("libket", "libket.dylib"), ("kbw", "libkbw.dylib")]
    raise OSError("unsupported operational system")


def _glibc_version() -> tuple[int, int] | None:
    """The glibc version on Linux, or ``None`` (other systems or libcs)."""
    if platform.system() != "Linux":
        return None
    name, version = platform.libc_ver()
    if name != "glibc" or not version:
        return None
    try:
        major, minor = version.split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return None


def _env_flag(name: str, default: bool) -> bool:
    """Reads a boolean environment variable; unset or empty gives ``default``."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be 1 or 0, got {os.environ[name]!r}")


def kbw_features() -> list[str]:
    """Cargo features to build KBW with.

    The ``gpu`` (``"dense gpu"`` and ``"sparse gpu"`` simulators) and ``cuda``
    (``"dense cuda"`` and ``"sparse cuda"``) features are enabled by default
    on Linux with glibc 2.34 or newer, as in the released wheels. The
    environment variables ``KET_BUILD_GPU`` and ``KET_BUILD_CUDA`` override
    this: ``1`` enables the feature and ``0`` disables it.
    """
    glibc = _glibc_version()
    default = glibc is not None and glibc >= (2, 34)
    return [feature for feature, env in _KBW_FEATURES if _env_flag(env, default)]


def make_libs():
    """Make Rust libs if missing.

    KBW is built with the features of :func:`kbw_features`.
    """
    dirname = os.path.dirname(__file__)

    if all(os.path.isfile(f"{dirname}/{lib}") for _, lib in libs()):
        return

    features = kbw_features()
    warnings.warn(
        "Compiling Libket and KBW"
        + (f" (features: {', '.join(features)})" if features else "")
        + "... may take a while. If it fails, check if Rust is installed "
        "https://www.rust-lang.org/tools/install."
    )

    build = ["cargo", "build", "--release", "--quiet"]
    subprocess.run(build, cwd=f"{dirname}/libket", check=True)
    kbw_build = build + (["--features", ",".join(features)] if features else [])
    subprocess.run(kbw_build, cwd=f"{dirname}/kbw", check=True)

    for dirlib, lib in libs():
        src = f"{dirname}/{dirlib}/target/release/{lib}"
        dst = f"{dirname}/{lib}"
        shutil.move(src, dst)


make_libs()
