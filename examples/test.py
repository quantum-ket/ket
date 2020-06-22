from ket import *
with run():
    q = qalloc(3)
    a = qalloc(3)
    m = measure(a)
    ket_temp_if_test0___ = m == 1
    if type(ket_temp_if_test0___) == future:
        ket_tmp_if_then1___ = label('if.then')
        ket_tmp_if_else1___ = label('if.else')
        ket_tmp_if_end1___ = label('if.end')
        branch(ket_temp_if_test0___, ket_tmp_if_then1___, ket_tmp_if_else1___)
        ket_tmp_if_then1___.begin()
        z(q)
        jump(ket_tmp_if_end1___)
        ket_tmp_if_else1___.begin()
        ket_temp_if_test0___ = m == 2
        ket_tmp_if_then0___ = label('if.then')
        ket_tmp_if_else0___ = label('if.else')
        ket_tmp_if_end0___ = label('if.end')
        branch(ket_temp_if_test0___, ket_tmp_if_then0___, ket_tmp_if_else0___)
        ket_tmp_if_then0___.begin()
        x(q)
        jump(ket_tmp_if_end0___)
        ket_tmp_if_else0___.begin()
        y(q)
        jump(ket_tmp_if_end0___)
        ket_tmp_if_end0___.begin()
        jump(ket_tmp_if_end1___)
        ket_tmp_if_end1___.begin()
    elif ket_temp_if_test0___:
        z(q)
    elif m == 2:
        x(q)
    else:
        y(q)
    print(measure(q).get())
