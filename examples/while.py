from ket import *
i = future(0)
count = future(0)
ket_temp_while_out_test0___ = i < 4
if type(ket_temp_while_out_test0___) == future:
    ket_tmp_while_test0___ = label('while.test')
    ket_tmp_while_loop0___ = label('while.loop')
    ket_tmp_while_end0___ = label('while.end')
    branch(ket_temp_while_out_test0___, ket_tmp_while_loop0___, ket_tmp_while_end0___)
    ket_tmp_while_test0___.begin()
    ket_temp_while_in_test0___ = i < 4
    branch(ket_temp_while_in_test0___, ket_tmp_while_loop0___, ket_tmp_while_end0___)
    ket_tmp_while_loop0___.begin()
    q = quant(3)
    h(q)
    i.set(measure(q))
    count.set(count + 1)
    jump(ket_tmp_while_test0___)
    ket_tmp_while_end0___.begin()
else:
    while i < 4:
        q = quant(3)
        h(q)
        i.set(measure(q))
        count.set(count + 1)
print(count.get())
