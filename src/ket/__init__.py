# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Ket quantum programming platform."""

from .clib import libs
from .base import *
from .base import __all__ as all_base
from .gates import *
from .gates import __all__ as all_gate

__version__ = "0.7.dev0"

__all__ = all_base + all_gate
