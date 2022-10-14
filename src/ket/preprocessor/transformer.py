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

import ast
import copy
# pylint: disable=missing-function-docstring, invalid-name


class ketpp (ast.NodeTransformer):
    """Ket interpreter preprocessor"""

    def __init__(self):
        self.id_count = 0
        self.while_begin = None
        self.while_end = None
        self.in_while = False

    def visit_If(self, node):

        if_id = self.id_count
        self.id_count += 1

        self.generic_visit(node)
        node_cp = copy.deepcopy(node)

        _test_name = '_ket_temp_if_test' + str(if_id)
        test_name = ast.Name(id=_test_name, ctx=ast.Load())

        end_name = '_ket_tmp_if_end' + str(if_id)
        else_name = None
        if node.orelse:
            else_name = '_ket_tmp_if_else' + str(if_id)
            if_begin = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[
                            ast.Name(id=else_name, ctx=ast.Store()),
                            ast.Name(id=end_name, ctx=ast.Store())
                        ],
                        ctx=ast.Store()
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id='_ket_if_else', ctx=ast.Load()),
                    args=[test_name],
                    keywords=[]
                )
            )
        else:
            if_begin = ast.Assign(
                targets=[
                    ast.Name(id=end_name, ctx=ast.Store()),
                ],
                value=ast.Call(
                    func=ast.Name(id='_ket_if', ctx=ast.Load()),
                    args=[test_name],
                    keywords=[]
                )
            )

        if_body = [if_begin, *node.body]

        if else_name is not None:
            else_call = ast.Expr(value=ast.Call(
                func=ast.Name(id='_ket_else', ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load())
                      for name in [else_name, end_name]],
                keywords=[]
            ))
            if_body.extend([else_call, *node.orelse])

        if_body.append(
            ast.Expr(value=ast.Call(
                func=ast.Name(id='_ket_next', ctx=ast.Load()),
                args=[ast.Name(id=end_name, ctx=ast.Load())],
                keywords=[]
            ))
        )

        test_assing = ast.Assign(
            targets=[ast.Name(id=_test_name, ctx=ast.Store())],
            value=node.test
        )

        type_check = ast.Call(
            func=ast.Name(id='_ket_is_future', ctx=ast.Load()),
            args=[test_name],
            keywords=[]
        )

        node_cp.test = test_name

        if_future = ast.If(
            test=type_check,
            body=if_body,
            orelse=[node_cp]
        )

        return [test_assing, if_future]

    def visit_While(self, node):  # pylint: disable=too-many-locals
        while_begin_save = self.while_begin
        while_end_save = self.while_end

        while_id = self.id_count
        self.id_count += 1

        begin_name = '_ket_tmp_while_begin' + str(while_id)
        loop_name = '_ket_tmp_while_loop' + str(while_id)
        end_name = '_ket_tmp_while_end' + str(while_id)
        else_name = None
        self.while_end = end_name
        if node.orelse:
            else_name = '_ket_tmp_while_else' + str(while_id)
            while_begin = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[ast.Name(id=name, ctx=ast.Store()) for name in [
                            begin_name, loop_name, else_name, end_name]],
                        ctx=ast.Store()
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id='_ket_while_else', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )
            )
        else:
            while_begin = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[ast.Name(id=name, ctx=ast.Store())
                              for name in [begin_name, loop_name, end_name]],
                        ctx=ast.Store()
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id='_ket_while', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )
            )

        test_name = '_ket_while_test' + str(while_id)
        test_assing = ast.Assign(
            targets=[ast.Name(id=test_name, ctx=ast.Store())],
            value=node.test
        )

        type_check = ast.Call(
            func=ast.Name(id='_ket_is_future', ctx=ast.Load()),
            args=[ast.Name(id=test_name, ctx=ast.Load())],
            keywords=[]
        )

        body_call = ast.Expr(value=ast.Call(
            func=ast.Name(id='_ket_while_body', ctx=ast.Load()),
            args=[
                ast.Name(id=test_name, ctx=ast.Load()),
                ast.Name(id=loop_name, ctx=ast.Load()),
                ast.Name(
                    id=else_name if else_name is not None else end_name, ctx=ast.Load())
            ],
            keywords=[]
        ))

        loop_call = ast.Expr(value=ast.Call(
            func=ast.Name(id='_ket_loop', ctx=ast.Load()),
            args=[
                ast.Name(id=begin_name, ctx=ast.Load()),
                ast.Name(
                    id=else_name if else_name is not None else end_name, ctx=ast.Load())
            ],
            keywords=[]
        ))

        else_call = [ast.Expr(value=ast.Call(
            func=ast.Name(id='_ket_next', ctx=ast.Load()),
            args=[ast.Name(id=end_name, ctx=ast.Load())],
            keywords=[]
        ))] if else_name is not None else []

        node_cp = copy.deepcopy(node)
        in_while_save = self.in_while
        self.in_while = True
        self.generic_visit(node)
        self.in_while = in_while_save

        self.generic_visit(node_cp)

        ndone_name = '_ket_tmp_while_ndone' + str(while_id)

        if_not_future = ast.If(
            test=ast.Name(id=test_name, ctx=ast.Load()),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=ndone_name, ctx=ast.Store())],
                    value=ast.Constant(value=True)
                ),
                ast.While(
                    test=ast.Name(id=ndone_name, ctx=ast.Load()),
                    body=[*node_cp.body,
                          ast.If(
                              test=ast.UnaryOp(
                                  op=ast.Not(), operand=node_cp.test),
                              body=[
                                  ast.Assign(
                                      targets=[
                                          ast.Name(id=ndone_name, ctx=ast.Store())],
                                      value=ast.Constant(value=False)
                                  )
                              ],
                              orelse=[]
                          )
                          ],
                    orelse=[]
                ),
                ast.If(
                    test=ast.UnaryOp(
                        op=ast.Not(),
                        operand=ast.Name(id=ndone_name, ctx=ast.Load())
                    ),
                    body=node_cp.orelse if node_cp.orelse else [ast.Pass()],
                    orelse=[]
                )
            ],
            orelse=node_cp.orelse
        )

        if_future = ast.If(
            test=type_check,
            body=[body_call, *node.body, loop_call, *else_call],
            orelse=[if_not_future]
        )

        self.while_begin = while_begin_save
        self.while_end = while_end_save

        return [while_begin, test_assing, if_future]

    def visit_Break(self, node):
        return ast.Expr(value=ast.Call(
            func=ast.Name(id='_ket_goto', ctx=ast.Load()),
            args=[ast.Name(id=self.while_end, ctx=ast.Load())],
            keywords=[]
        )) if self.in_while else node

    def visit_Continue(self, node):
        return ast.Expr(value=ast.Call(
            func=ast.Name(id='_ket_goto', ctx=ast.Load()),
            args=[ast.Name(id=self.while_begin, ctx=ast.Load())],
            keywords=[]
        )) if self.in_while else node
