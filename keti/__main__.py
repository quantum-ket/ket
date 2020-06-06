import ast
import argparse
from .pp import ketpp
import sys
from ast_decompiler import decompile

def main():

    parser = argparse.ArgumentParser(description='Ket interprester')
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
