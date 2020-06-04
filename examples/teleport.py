from ket import *

def bell(aux0, aux1):
    q = alloc(2)
    if type(aux0) == future:
        ket_tmp_if_then0___ = label('if.then')
        ket_tmp_if_end0___ = label('if.end')
        branch(aux0, ket_tmp_if_then0___, ket_tmp_if_end0___)
        ket_tmp_if_then0___.begin()
        x(q(0))
        jump(ket_tmp_if_end0___)
        ket_tmp_if_end0___.begin()
    elif aux0:
        x(q(0))
    if type(aux1) == future:
        ket_tmp_if_then1___ = label('if.then')
        ket_tmp_if_end1___ = label('if.end')
        branch(aux1, ket_tmp_if_then1___, ket_tmp_if_end1___)
        ket_tmp_if_then1___.begin()
        x(q(1))
        jump(ket_tmp_if_end1___)
        ket_tmp_if_end1___.begin()
    elif aux1:
        x(q(1))
    h(q(0))
    ctrl(q(0), x, q(1))
    return q

def teleport(a):
    b = bell(0, 0)
    ctrl(a, x, b(0))
    h(a)
    m0 = measure(a)
    m1 = measure(b(0))
    if type(m1) == future:
        ket_tmp_if_then2___ = label('if.then')
        ket_tmp_if_end2___ = label('if.end')
        branch(m1, ket_tmp_if_then2___, ket_tmp_if_end2___)
        ket_tmp_if_then2___.begin()
        x(b(1))
        jump(ket_tmp_if_end2___)
        ket_tmp_if_end2___.begin()
    elif m1:
        x(b(1))
    if type(m0) == future:
        ket_tmp_if_then3___ = label('if.then')
        ket_tmp_if_end3___ = label('if.end')
        branch(m0, ket_tmp_if_then3___, ket_tmp_if_end3___)
        ket_tmp_if_then3___.begin()
        z(b(1))
        jump(ket_tmp_if_end3___)
        ket_tmp_if_end3___.begin()
    elif m0:
        z(b(1))
    return b(1)
a = alloc(1)
h(a)
z(a)
y = teleport(a)
h(y)
print(measure(y).get())
