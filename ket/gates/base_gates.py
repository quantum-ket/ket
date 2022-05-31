from ..libket import ctrl_push, ctrl_pop, X, Z
from functools import reduce
from operator import add


def _flipc(state, q):
    length = len(q)
    if hasattr(state, '__iter__'):
        if len(state) != length:
            raise ValueError(
                f"'to' received a list of length {len(state)} to use on {length} qubits")
    else:
        if length < state.bit_length():
            raise ValueError(
                f"To flip with {state=} you need at least {state.bit_length()} qubits")
        state = [int(i) for i in f"{{:0{length}b}}".format(state)]
    for i, q in zip(state, q):
        if i == 0:
            X(q)


def _phase_on(state, q):
    q = reduce(add, q)
    _flipc(state, q)
    ctrl_push(q[1:])
    Z(q[0])
    ctrl_pop()
    _flipc(state, q)
