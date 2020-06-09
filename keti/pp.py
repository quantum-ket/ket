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

class ketpp (ast.NodeTransformer):

    def __init__(self):
        self.id_count = 0

    def visit_If(self, node):
        self.generic_visit(node)

        ########### LABEL #############
        label_call = lambda name, label_id : ast.Call(func=ast.Name(id='label', ctx=ast.Load()), args=[ast.Constant(value=label_id, kind=None)], keywords=[])
        label_assing = lambda name, label_id : ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=label_call(name, label_id))
        then_name = 'ket_tmp_if_then'+str(self.id_count)+'___'
        then_label_stmt = label_assing(then_name, 'if.then')
        end_name = 'ket_tmp_if_end'+str(self.id_count)+'___'
        end_label_stmt = label_assing(end_name, 'if.end')
        if len(node.orelse) != 0:
            else_name = 'ket_tmp_if_else'+str(self.id_count)+'___' 
            else_label_stmt = label_assing(else_name, 'if.else')
            if_body = [then_label_stmt, else_label_stmt, end_label_stmt]
            branch_call = ast.Call(func=ast.Name(id='branch', ctx=ast.Load()), args=[node.test, ast.Name(id=then_name, ctx=ast.Load()), ast.Name(id=else_name, ctx=ast.Load())], keywords=[])
        else:
            branch_call = ast.Call(func=ast.Name(id='branch', ctx=ast.Load()), args=[node.test, ast.Name(id=then_name, ctx=ast.Load()), ast.Name(id=end_name, ctx=ast.Load())], keywords=[])
            if_body = [then_label_stmt, end_label_stmt]
        
        self.id_count += 1
        
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
