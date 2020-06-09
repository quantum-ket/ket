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

import ast
import argparse
from .pp import ketpp
import sys
from ast_decompiler import decompile

def main():

    parser = argparse.ArgumentParser(prog='ket', description='Ket interprester')
    parser.add_argument('input', metavar='.ket', nargs=argparse.REMAINDER, type=str, help='source code')
    parser.add_argument('-e', '--preprocessor', metavar='.py', help='emit preprocessor output')

    args = parser.parse_args()

    if len(args.input) == 0:
        print("No input")
        exit()

    with open(args.input[0], 'r') as source:
        tree = ast.parse(source.read())

    tree.body.insert(0, ast.ImportFrom(module='ket', names=[ast.alias(name='*', asname=None)], level=0))

    pp = ketpp()

    pp.visit(tree)
    ast.fix_missing_locations(tree)

    if args.preprocessor != None:
        with open(args.preprocessor, 'w') as out:
            out.write(decompile(tree))

    obj = compile(tree, args.input[0], 'exec', optimize=2)

    sys.argv = args.input

    exec(obj, globals())

if __name__ == '__main__':
    main()
