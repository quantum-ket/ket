# MIT License
# 
# Copyright (c) 2020 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
# Copyright (c) 2020 Rafael de Santiago <r.santiago@ufsc.br>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .import_ket import __import_module_ket__, __import_from_ket__
from ket import *

def __ket__():
    import argparse
    from .import_ket import __import_ket__
    from os import path, getcwd

    parser = argparse.ArgumentParser(prog='ket', description='Ket interprester')
    parser.add_argument('input', metavar='.ket', nargs=argparse.REMAINDER, type=str, help='source code')

    args = parser.parse_args()

    if len(args.input) == 0:
        print("No input")
        exit()
        
    workdir = path.dirname(path.join(getcwd(), args.input[0]))
    return __import_ket__(path.basename(args.input[0]), workdir)

if __name__ == '__main__':
    exec(__ket__(), globals())
