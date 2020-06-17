from ket import *

def bell(aux0, aux1):
    q = qalloc(2)
    ket_temp_if_test0___ = aux0 == 1
    if type(ket_temp_if_test0___) == future:
        ket_tmp_if_then0___ = label('if.then')
        ket_tmp_if_end0___ = label('if.end')
        branch(ket_temp_if_test0___, ket_tmp_if_then0___, ket_tmp_if_end0___)
        ket_tmp_if_then0___.begin()
        x(q(0))
        jump(ket_tmp_if_end0___)
        ket_tmp_if_end0___.begin()
    elif ket_temp_if_test0___:
        x(q(0))
    ket_temp_if_test1___ = aux1 == 1
    if type(ket_temp_if_test1___) == future:
        ket_tmp_if_then1___ = label('if.then')
        ket_tmp_if_end1___ = label('if.end')
        branch(ket_temp_if_test1___, ket_tmp_if_then1___, ket_tmp_if_end1___)
        ket_tmp_if_then1___.begin()
        x(q(1))
        jump(ket_tmp_if_end1___)
        ket_tmp_if_end1___.begin()
    elif ket_temp_if_test1___:
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
    ket_temp_if_test2___ = m1 == 1
    if type(ket_temp_if_test2___) == future:
        ket_tmp_if_then2___ = label('if.then')
        ket_tmp_if_end2___ = label('if.end')
        branch(ket_temp_if_test2___, ket_tmp_if_then2___, ket_tmp_if_end2___)
        ket_tmp_if_then2___.begin()
        x(b(1))
        jump(ket_tmp_if_end2___)
        ket_tmp_if_end2___.begin()
    elif ket_temp_if_test2___:
        x(b(1))
    ket_temp_if_test3___ = m0 == 1
    if type(ket_temp_if_test3___) == future:
        ket_tmp_if_then3___ = label('if.then')
        ket_tmp_if_end3___ = label('if.end')
        branch(ket_temp_if_test3___, ket_tmp_if_then3___, ket_tmp_if_end3___)
        ket_tmp_if_then3___.begin()
        z(b(1))
        jump(ket_tmp_if_end3___)
        ket_tmp_if_end3___.begin()
    elif ket_temp_if_test3___:
        z(b(1))
    return b(1)
a = qalloc(1)
h(a)
z(a)
y = teleport(a)
h(y)
print(measure(y).get())
