import ast
from sys import argv

class ket_pp (ast.NodeTransformer):

    def visit_If(self, node):
        self.generic_visit(node)

        ########### LABEL #############
        label_call = lambda name : ast.Call(func=ast.Name(id='label', ctx=ast.Load()), args=[ast.Constant(value=name, kind=None)], keywords=[])
        label_assing = lambda name : ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=label_call(name))
        then_name = 'ket_tmp_if_then___'
        then_label_stmt = label_assing(then_name)
        end_name = 'ket_tmp_if_end___'
        end_label_stmt = label_assing(end_name)
        if len(node.orelse) != 0:
            else_name = 'ket_tmp_if_else___' 
            else_label_stmt = label_assing(else_name)
            if_body = [then_label_stmt, else_label_stmt, end_label_stmt]
            branch_call = ast.Call(func=ast.Name(id='branch', ctx=ast.Load()), args=[node.test, ast.Name(id=then_name, ctx=ast.Load()), ast.Name(id=else_name, ctx=ast.Load())], keywords=[])
        else:
            branch_call = ast.Call(func=ast.Name(id='branch', ctx=ast.Load()), args=[node.test, ast.Name(id=then_name, ctx=ast.Load()), ast.Name(id=end_name, ctx=ast.Load())], keywords=[])
            if_body = [then_label_stmt, end_label_stmt]
        
        branch_stmt = ast.Expr(branch_call)

        ########### THEN #############
        label_begin = lambda name : ast.Call(func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr='begin', ctx=ast.Load()), args=[], keywords=[])
        then_begin_stmt = ast.Expr(label_begin(then_name))

        if_body.extend([branch_stmt, then_begin_stmt])
        if_body.extend(node.body)

        jump_call = ast.Call(func=ast.Name(id='jump', ctx=ast.Load()), args=[ast.Name(id=end_name, ctx=ast.Load())], keywords=[])
        jump_stmt = ast.Expr(jump_call)

        if_body.append(jump_stmt)

        ########### ELSE #############
        if len(node.orelse) != 0:
            else_begin_stmt = ast.Expr(label_begin(else_name))
            if_body.append(else_begin_stmt)
            if_body.extend(node.orelse)
            if_body.append(jump_stmt)

        end_begin_stmt = ast.Expr(label_begin(end_name))
        
        if_body.append(end_begin_stmt)

        ##############################

        type_check = 'future'
        type_call_exp = ast.Call(func=ast.Name(id='type', ctx=ast.Load()), args=[node.test], keywords=[])
        type_eq_int_exp = ast.Compare(left=type_call_exp, ops=[ast.Eq()], comparators=[ast.Name(id=type_check, ctx=ast.Load())])

        if_future_stmt = ast.If(test=type_eq_int_exp, body=if_body, orelse=[node])

        return [if_future_stmt]

def main():
    with open(argv[1], 'r') as source:
        tree = ast.parse(source.read())

    tree.body.insert(0, ast.ImportFrom(module='ket', names=[ast.alias(name='*', asname=None)], level=0))

    pp = ket_pp()

    pp.visit(tree)
    ast.fix_missing_locations(tree)

    obj = compile(tree, argv[1], 'exec', optimize=2)
    exec(obj, globals())

if __name__ == '__main__':
    main()
