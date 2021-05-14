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

from .import_ket import __import_globals_ket__
from . import *
from .ket import label, branch, jump

def __ket__():
    import argparse
    from os import path, getcwd

    parser = argparse.ArgumentParser(prog='ket', description='Ket interprester')
    parser.add_argument('input', metavar='.ket', nargs=argparse.REMAINDER, type=str, help='source code')

    args = parser.parse_args()

    if len(args.input) == 0:
        print("No input")
        exit(1)

    globals()['__name__'] = '__main__'
    globals()['__in_ket__'] = True
    source = path.join(getcwd(), args.input[0])
    __import_globals_ket__(source, globals())

if __name__ == '__main__':
    __ket__()
