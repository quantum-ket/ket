from .. import plugin, quant

def pown (a : int, x : quant, n : int) -> quant:
    """Apply a modular exponentiation in a superposition.

        .. math::

            \left|x\\right>\left|0\\right> \\rightarrow  \left|x\\right>\left|a^x\;\\text{mod}\,n\\right>                 

        Return:
            Quant with the result of the operation. 
    """
    
    ret = quant(len(x))

    tmp = str(int(n)) + ' ' + str(int(a))
    plugin('ket_pown', x|ret, tmp)

    return ret
    