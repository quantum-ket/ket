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

from .import_ket import _import_globals_ket
from . import *
from .ket import label, branch, jump, build_info
from .preprocessor import *

def __ket__():
    import argparse
    from os import path, getcwd

    parser = argparse.ArgumentParser(prog='ket', description='Ket interpreter')
    parser.add_argument('--version', action='version', version=f'Ket {build_info()}')

    parser.add_argument('-o', '--out',      help='KQASM output file',                             type=str)
    parser.add_argument('-s', '--kbw',      help='quantum execution (KBW) IP address',            type=str, default='127.0.0.1')
    parser.add_argument('-u', '--user',     help='quantum execution (KBW) SSH user',              type=str)    
    parser.add_argument('-p', '--port',     help='quantum execution (KBW) port',                  type=str, default='4242')
    parser.add_argument('-P', '--ssh-port', help='quantum execution (KBW) SSH port',              type=str, default='22')
    parser.add_argument('--seed',           help='set RNG seed for quantum execution',            type=int)
    parser.add_argument('--api-args',       help='additional parameters for quantum execution',   type=str)
    parser.add_argument('--no-execute',     help='does not execute KQASM, measurements return 0', action='store_false')
    parser.add_argument('--dump2fs',        help='use the filesystem to transfer dump data',      action='store_true')
    parser.add_argument('input',            metavar='.ket',      help='source code',                                   type=str)

    args = parser.parse_args()

    ket_args = {
        "server"   : args.kbw, 
        "port"     : args.port, 
        "execute"  : args.no_execute,
        "dump2fs"  : args.dump2fs,
        "ssh_port" : args.ssh_port,
    }

    if args.user:
        ket_args["user"] = args.user
    if args.out:
        ket_args["kqasm"] = args.out
    if args.seed:
        ket_args["seed"] = args.seed
    if args.api_args:
        ket_args["api-args"] = args.api_args
    
    ket_config(**ket_args)

    globals()['__name__'] = '__main__'
    globals()['__in_ket__'] = True
    source = path.join(getcwd(), args.input)
    _import_globals_ket(source, globals())

if __name__ == '__main__':
    __ket__()
