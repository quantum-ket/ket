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

import ast, copy

class ketpp (ast.NodeTransformer):

    def __init__(self, workdir : str):
        self.id_count = 0
        self.in_if = False
        self.in_while = False
        self.workdir = workdir

    def label_new(self, name, label_id):
        label_call = ast.Call(func=ast.Name(id='label', ctx=ast.Load()), args=[ast.Constant(value=label_id, kind=None)], keywords=[])
        return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=label_call)
    
    def label_begin(self, name):
        call = ast.Call(func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr='begin', ctx=ast.Load()), args=[], keywords=[])
        return ast.Expr(call)

    def branch_call(self, test_name, true_name, false_name):
        call = ast.Call(func=ast.Name(id='branch', ctx=ast.Load()), args=[ast.Name(id=test_name, ctx=ast.Load()), ast.Name(id=true_name, ctx=ast.Load()), ast.Name(id=false_name, ctx=ast.Load())], keywords=[])
        return ast.Expr(call)

    def jump_call(self, label_name):
        call =  ast.Call(func=ast.Name(id='jump', ctx=ast.Load()), args=[ast.Name(id=label_name, ctx=ast.Load())], keywords=[])
        return ast.Expr(call)
    
    def type_future(self, test_name):
        type_call_exp = ast.Call(func=ast.Name(id='type', ctx=ast.Load()), args=[ast.Name(id=test_name, ctx=ast.Load())], keywords=[])
        return ast.Compare(left=type_call_exp, ops=[ast.Eq()], comparators=[ast.Name(id='future', ctx=ast.Load())])

    def visit_If(self, node):
        tmp = self.in_if
        self.in_if = True
        node_copy = copy.deepcopy(node)
        self.generic_visit(node)
        self.in_if = tmp
 
        test_name = 'ket_temp_if_test'+str(self.id_count)+'___' 
        test_assing = ast.Assign(targets=[ast.Name(id=test_name, ctx=ast.Store())], value=node.test)

        ########### LABEL #############

        then_name = 'ket_tmp_if_then'+str(self.id_count)+'___'
        then_label = self.label_new(then_name, 'if.then')

        end_name = 'ket_tmp_if_end'+str(self.id_count)+'___'
        end_label = self.label_new(end_name, 'if.end')

        if node.orelse:
            else_name = 'ket_tmp_if_else'+str(self.id_count)+'___' 
            else_label = self.label_new(else_name, 'if.else')
            if_body = [then_label, else_label, end_label]

            branch_stmt = self.branch_call(test_name, then_name, else_name)
        else:
            branch_stmt = self.branch_call(test_name, then_name, end_name)
            if_body = [then_label, end_label]
        
        ########### THEN #############

        then_begin = self.label_begin(then_name)

        if_body.extend([branch_stmt, then_begin])
        if_body.extend(node.body)

        jump_stmt = self.jump_call(end_name)

        if_body.append(jump_stmt)

        ########### ELSE #############

        if node.orelse:
            else_begin = self.label_begin(else_name)
            if_body.append(else_begin)
            if_body.extend(node.orelse)
            if_body.append(jump_stmt)

        end_begin = self.label_begin(end_name)
        
        if_body.append(end_begin)

        ##############################

        self.id_count += 1
        
        if not self.in_if:
            test_type = self.type_future(test_name)

            node_copy.test = ast.Name(id=test_name, ctx=ast.Load())
            if_future_stmt = ast.If(test=test_type, body=if_body, orelse=[node_copy])

            return [test_assing, if_future_stmt]
        else:
            ret_body = [test_assing]
            ret_body.extend(if_body)

            return ret_body

    def visit_While(self, node):
        tmp = self.in_while
        self.in_while = True
        node_copy = copy.deepcopy(node)
        self.generic_visit(node)
        self.in_while = tmp

        test_out_name = 'ket_temp_while_out_test'+str(self.id_count)+'___' 
        test_out_assing = ast.Assign(targets=[ast.Name(id=test_out_name, ctx=ast.Store())], value=node.test)

        ########### LABEL #############

        test_name = 'ket_tmp_while_test'+str(self.id_count)+'___'
        test_new = self.label_new(test_name, 'while.test')
        
        loop_name = 'ket_tmp_while_loop'+str(self.id_count)+'___'
        loop_new = self.label_new(loop_name, 'while.loop')
        
        end_name = 'ket_tmp_while_end'+str(self.id_count)+'___'
        end_new = self.label_new(end_name, 'while.end')
        
        if_body = [test_new, loop_new, end_new]
        if node.orelse:
            else_name = 'ket_tmp_while_else'+str(self.id_count)+'___'
            else_new = self.label_new(else_name, 'while.else')
            if_body.append(else_new)

        ########### TEST #############
        
        if node.orelse: 
            branch_stmt = self.branch_call(test_out_name, loop_name, else_name)
        else:
            branch_stmt = self.branch_call(test_out_name, loop_name, end_name)

        test_begin = self.label_begin(test_name)
        
        test_in_name = 'ket_temp_while_in_test'+str(self.id_count)+'___' 
        test_in_assing = ast.Assign(targets=[ast.Name(id=test_in_name, ctx=ast.Store())], value=node.test)
        
        if node.orelse: 
            branch_in_stmt = self.branch_call(test_in_name, loop_name, else_name)
        else:
            branch_in_stmt = self.branch_call(test_in_name, loop_name, end_name)

        ########### LOOP #############

        loop_begin = self.label_begin(loop_name)

        if_body.extend([branch_stmt, test_begin, test_in_assing, branch_in_stmt, loop_begin])
        if_body.extend(node.body)

        jump_stmt = self.jump_call(test_name)

        if_body.append(jump_stmt)

        ########### ELSE #############

        if node.orelse:
            else_begin = self.label_begin(else_name)
            
            if_body.append(else_begin)
            if_body.extend(node.orelse)

            jump_end = self.jump_call(end_name)
            
            if_body.append(jump_end)
            
        end_begin =  self.label_begin(end_name)

        if_body.append(end_begin)

        ########### IF #############
        
        test_out_type = self.type_future(test_out_name)

        if_future = ast.If(test=test_out_type, body=if_body, orelse=[node_copy])

        self.id_count += 1
        return [test_out_assing, if_future]

    def visit_Break(self, node):
        if self.in_while:
            continuing_name = 'ket_tmp_continuing'+str(self.id_count)+'___'
            continue_label = self.label_new(continuing_name, 'continuing')
            end_name = 'ket_tmp_while_end'+str(self.id_count)+'___'
            jump = self.jump_call(end_name)
            continue_begin = self.label_begin(continuing_name)
            return [continue_label, jump, continue_begin]
        else:
            return node
         
    def visit_Continue(self, node):
        if self.in_while:
            continuing_name = 'ket_tmp_continuing'+str(self.id_count)+'___'
            continue_label = self.label_new(continuing_name, 'continuing')
            test_name = 'ket_tmp_while_test'+str(self.id_count)+'___'
            jump = self.jump_call(test_name)
            continue_begin = self.label_begin(continuing_name)
            return [continue_label, jump, continue_begin]
        else:
            return node

    def visit_Import(self, node):
        import_list = []
        ket_import_list = []
        for name in node.names:
            if name.name[-4:] == '.ket':
                if name.asname:
                    module_name = name.asname
                else:
                    module_name = name.name[:-4]
                import_call = ast.Call(func=ast.Name(id='__import_module_ket__', ctx=ast.Load()), args=[ast.Constant(value=name.name, kind=None), ast.Constant(value=self.workdir, kind=None) ], keywords=[])
                ket_import_list.append(ast.Assign(targets=[ast.Name(id=module_name, ctx=ast.Store())], value=import_call))
            else:
                import_list.append(name)
        if import_list:
            ket_import_list.append(ast.Import(import_list))
        return ket_import_list
    
    def visit_ImportFrom(self, node):
        if node.module[-4:] == '.ket':
            ret_list = []
            import_call = ast.Call(func=ast.Name(id='__import_module_ket__', ctx=ast.Load()), args=[ast.Constant(value=node.module, kind=None), ast.Constant(value=self.workdir, kind=None) ], keywords=[])
            for name in node.names:
                if name.asname:
                    to_name = name.asname
                else:
                    to_name = name.name
                element  = ast.Attribute(value=import_call, attr=name.name, ctx=ast.Load())
                ret_list.append(ast.Assign(targets=[ast.Name(id=to_name, ctx=ast.Store())], value=element))
            return ret_list
        else:
            return node
