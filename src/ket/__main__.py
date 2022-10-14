#  Copyright 2020, 2021 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
#  Copyright 2020, 2021 Rafael de Santiago <r.santiago@ufsc.br>
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

# pylint: disable=unused-import, wildcard-import

from .import_ket import _import_globals_ket
from . import *  # noqa: F403, F401
from . import __version__
from .base import label, branch, jump  # noqa: F401
from .preprocessor.statements import *  # noqa: F403, F401


def __ket__():
    import argparse  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(prog='ket', description='Ket interpreter')
    parser.add_argument('--version', action='version',
                        version=f'Ket {__version__}')

    parser.add_argument('input', metavar='.ket',
                        help='source code', type=str)

    args = parser.parse_args()

    globals()['__name__'] = '__main__'
    globals()['__in_ket__'] = True
    source = os.path.join(os.getcwd(), args.input)
    _import_globals_ket(source, globals())


if __name__ == '__main__':
    __ket__()
