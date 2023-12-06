# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from .clib import libs
from .gates import *
from .import_ket import *
from .base import *
from .standard import *
from .process import *
from .gates import __all__ as all_gate
from .import_ket import __all__ as all_import
from .base import __all__ as all_base
from .standard import __all__ as all_standard
from .process import __all__ as all_process

__version__ = '0.6.1'
__all__ = all_gate + all_import + all_base + all_standard + all_process

from .import_ket import code_ket

from .base import QUANTUM_EXECUTION_TARGET
if QUANTUM_EXECUTION_TARGET is None:
    from .kbw import use_sparse
    use_sparse()
