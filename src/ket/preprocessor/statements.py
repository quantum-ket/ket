from __future__ import annotations
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

from ..base import label, branch, jump, future


__all__ = ['_ket_is_future', '_ket_if', '_ket_if_else', '_ket_else', '_ket_next',
           '_ket_while', '_ket_while_else', '_ket_while_body', '_ket_loop', '_ket_goto']


def _ket_is_future(obj) -> bool:
    return isinstance(obj, future)


def _ket_if(test: future) -> label:
    if_then = label()
    if_end = label()
    branch(test, if_then, if_end)
    if_then.begin()
    return if_end


def _ket_if_else(test: future) -> tuple[label]:
    if_then = label()
    if_else = label()
    if_end = label()
    branch(test, if_then, if_else)
    if_then.begin()
    return if_else, if_end


def _ket_else(else_: label, end: label):
    jump(end)
    else_.begin()


def _ket_next(end: label):
    jump(end)
    end.begin()


def _ket_while() -> tuple[label]:
    while_test = label()
    while_loop = label()
    while_end = label()
    jump(while_test)
    while_test.begin()
    return while_test, while_loop, while_end


def _ket_while_else() -> tuple[label]:
    while_test = label()
    while_loop = label()
    while_else = label()
    while_end = label()
    jump(while_test)
    while_test.begin()
    return while_test, while_loop, while_else, while_end


def _ket_while_body(test: future, body: label, end: label):
    branch(test, body, end)
    body.begin()


def _ket_loop(test: label, end: label):
    jump(test)
    end.begin()


def _ket_goto(where: label):
    dead_code = label()
    jump(where)
    dead_code.begin()
