from . import plugin, quant

def pown (a : int, x : quant, n : int):
    ret = quant(len(x))

    tmp = str(int(n)) + ' ' + str(int(a))
    plugin('ket_pown', x|ret, tmp)

    return ret
    