#Licensed under the Apache License, Version 2.0;
#Copyright 2022 Evandro Chagas Ribeiro da Rosa
from math import pi
from ctypes import *
from typing import Iterable, Optional
from random import randint
from os import environ
from os.path import dirname

__all__ = ['quant', 'future', 'dump']

def load_libketc():
    if "LIBKETC_PATH" in environ:
        libketc_path = environ["LIBKETC_PATH"]
    else:
        libketc_path = dirname(__file__)+"/libketc.so"

    return cdll.LoadLibrary(libketc_path)

def set_kbw_path():
    if "KET_QUANTUM_EXECUTOR" not in environ:
        environ["KET_QUANTUM_EXECUTOR"] = dirname(__file__)+"/kbw.so"
    if "KQE_PLUGIN_PATH" not in environ:
        environ["KQE_PLUGIN_PATH"] = dirname(__file__)

class libket_error(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

KET_SUCCESS    = 0
KET_ERROR      = 1
KET_PAULI_X    = 2 
KET_PAULI_Y    = 3   
KET_PAULI_Z    = 4
KET_ROTATION_X = 5
KET_ROTATION_Y = 6
KET_ROTATION_Z = 7
KET_HADAMARD   = 8
KET_PHASE      = 9
KET_INT_EQ     = 10
KET_INT_NEQ    = 11
KET_INT_GT     = 12
KET_INT_GEQ    = 13
KET_INT_LT     = 14
KET_INT_LEQ    = 15
KET_INT_ADD    = 16
KET_INT_SUB    = 17
KET_INT_MUL    = 18
KET_INT_DIV    = 19
KET_INT_SLL    = 20
KET_INT_SRL    = 21
KET_INT_AND    = 22
KET_INT_OR     = 23
KET_INT_XOR    = 24
KET_INT_FF     = 25
KET_INT_FI     = 26
KET_INT_IF     = 27

set_kbw_path()

libketc = load_libketc()
ket_error_message = libketc.ket_error_message
ket_error_message.argtypes = []
ket_error_message.restype = c_char_p

def ket_error_warpper(error : c_int):
    if error == KET_ERROR:
        raise libket_error(str(ket_error_message().decode()))

class qubit:
    ket_qubit_new = libketc.ket_qubit_new
    ket_qubit_new.argtypes = [POINTER(c_void_p)]

    ket_qubit_delete = libketc.ket_qubit_delete
    ket_qubit_delete.argtypes = [c_void_p]

    ket_qubit_index = libketc.ket_qubit_index
    ket_qubit_index.argtypes = [c_void_p, POINTER(c_uint)]

    ket_qubit_measured = libketc.ket_qubit_measured
    ket_qubit_measured.argtypes = [c_void_p, POINTER(c_bool)]

    ket_qubit_allocated = libketc.ket_qubit_allocated
    ket_qubit_allocated.argtypes = [c_void_p, POINTER(c_bool)]

    ket_qubit_process_id = libketc.ket_qubit_process_id
    ket_qubit_process_id.argtypes = [c_void_p, POINTER(c_uint)]

    def __init__(self):
        self._as_parameter_ = c_void_p()
        ket_error_warpper(
            self.ket_qubit_new(byref(self._as_parameter_))
        )

    def __del__(self):
        ket_error_warpper(
            self.ket_qubit_delete(self)
        )

    @property
    def index(self):
        c_value = c_uint()
        ket_error_warpper(
            self.ket_qubit_index(self, c_value)
        )
        return c_value.value

    @property
    def measured(self):
        c_value = c_bool()
        ket_error_warpper(
            self.ket_qubit_measured(self, c_value)
        )
        return c_value.value

    @property
    def allocated(self):
        c_value = c_bool()
        ket_error_warpper(
            self.ket_qubit_allocated(self, c_value)
        )
        return c_value.value

    @property
    def process_id(self):
        c_value = c_uint()
        ket_error_warpper(
            self.ket_qubit_process_id(self, c_value)
        )
        return c_value.value

    def __repr__(self):
        return f"<Ket 'qubit' {(self.process_id, self.index)}>"

class quant:
    r"""Qubit list

    Allocate ``size`` qubits in the state :math:`\left|0\right>` and return
    its reference in a new :class:`~ket.types.quant`.

    Qubits allocated using a ``with`` statement must be released at the end of
    the scope.
    
    Example:
    
    .. code-block:: ket

        a = H(quant()) 
        b = X(quant())
        with quant() as aux: 
            with around(H, aux):
                with control(aux):
                    swap(a, b)
            result = measure(aux)
            if result == 1:
                X(aux) 
            aux.free() 

    :Qubit Indexing:

    Use brackets to index qubits as in a ``list`` and use ``+`` to concatenate
    two :class:`~ket.types.quant`.

    Example:

    .. code-block:: ket

        q = quant(20)        
        head, tail = q[0], q[1:]
        init, last = q[:-1], q[-1]
        even = q[::2]
        odd = q[1::2]
        reverse = reversed(q) # invert qubits order

        
        a, b = quant(2) # |a⟩, |b⟩
        c = a+b         # |c⟩ = |a⟩|b⟩ 

    warning:
        Use dirty qubits may have side effects due to previous entanglement.

    Args:
        size: The number of qubits to allocate.
        dirty: If `True`, Allocate ``size`` qubits at an unknown state.
    """

    def __init__(self, size : int = 1, dirty : bool = False, *, qubits = None):
        if qubits:
            self.qubits = qubits
        else:
            self.qubits = [process_top().alloc(dirty) for _ in range(size)]

    def __add__(self, other):
        return quant(qubits=self.qubits+other.qubits)

    def at(self, index : Iterable[int]):
        r"""Return qubits at `index`
        
        Create a new :class:`~ket.types.quant` with the qubit references at the
        position defined by the ``index`` list.

        :Example:

        .. code-block:: ket

            q = quant(20)        
            odd = q.at(range(1, len(q), 2)) # = q[1::2]

        Args:
            index: List of indexes.
        """

        return quant(qubits=[self.qubits[i] for i in index])
    
    def free(self, dirty : bool = False):
        r"""Free the qubits

        All qubits must be at the state :math:`\left|0\right>` before the call,
        otherwise set the ``dirty`` param to ``True``.

        Warning: 
            No check is applied to see if the qubits are at state
            :math:`\left|0\right>`.

        Args:
            dirty: Set ``True`` to free dirty qubits.
        """
 
        for qubit in self.qubits:
            process_top().free(qubit)            
        
    def is_free(self) -> bool:
        """Return ``True`` when all qubits are free"""
        return all(not qubit.allocated for qubit in self.qubits)

    def __reversed__(self):
        return quant(qubits=reversed(self.qubits))

    def __getitem__(self, key):
        qubits = self.qubits.__getitem__(key)
        return quant(qubits=qubits if isinstance(qubits, list) else [qubits])

    class iter:
        def __init__(self, q):
            self.q = q
            self.idx = -1
            self.size = len(q.qubits)

        def __next__(self): 
            self.idx += 1
            if self.idx < self.size:
                return self.q[self.idx]
            raise StopIteration

        def __iter__(self):
            return self
 
    def __iter__(self):
        return self.iter(self)

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        if not self.is_free():
            raise RuntimeError('non-free quant at the end of scope')

    def __len__(self):
        return len(self.qubits)

    def __repr__(self):
        if len(self.qubits) > 5:
            qubits_str = ', '.join(repr(self.qubits[i]) for i in range(5))+', ...'
        else:
            qubits_str = ', '.join(repr(q) for q in self.qubits)

        return f"<Ket 'quant' ({qubits_str})>"
        
class dump:
    ket_dump_new = libketc.ket_dump_new
    ket_dump_new.argtypes = [POINTER(c_void_p)]

    ket_dump_delete = libketc.ket_dump_delete
    ket_dump_delete.argtypes = [c_void_p]

    ket_dump_states = libketc.ket_dump_states
    ket_dump_states.argtypes = [c_void_p, POINTER(c_void_p), POINTER(c_size_t)]

    ket_dump_state_at = libketc.ket_dump_state_at
    ket_dump_state_at.argtypes = [c_void_p, POINTER(POINTER(c_ulong)), POINTER(c_size_t), c_ulong]

    ket_dump_amplitudes = libketc.ket_dump_amplitudes
    ket_dump_amplitudes.argtypes = [c_void_p, POINTER(c_void_p), POINTER(c_size_t)]
    
    ket_dump_amp_at = libketc.ket_dump_amp_at
    ket_dump_amp_at.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double), c_ulong]

    ket_dump_available = libketc.ket_dump_available
    ket_dump_available.argtypes = [c_void_p, POINTER(c_bool)]

    ket_dump_index = libketc.ket_dump_index
    ket_dump_index.argtypes = [c_void_p, POINTER(c_uint)]

    ket_dump_process_id = libketc.ket_dump_process_id
    ket_dump_process_id.argtypes = [c_void_p, POINTER(c_uint)]

    def __init__(self, q : quant):
        self._as_parameter_ = c_void_p()
        ket_error_warpper(
            self.ket_dump_new(byref(self._as_parameter_))
        )
        process_top().dump(self, *q.qubits)
        self.size = len(q.qubits)

    def __del__(self):
        ket_error_warpper(
            self.ket_dump_delete(self)
        )
    
    @property
    def states(self):
        if not self.available:
            exec_quantum()

        c_states = c_void_p()
        c_size   = c_size_t()
        ket_error_warpper(
            self.ket_dump_states(self, c_states, c_size)
        )
        for i in range(c_size.value):
            c_state = POINTER(c_ulong)()
            c_state_size = c_size_t()
            ket_error_warpper(
                self.ket_dump_state_at(c_states, c_state, c_state_size, i)
            )
            yield int(''.join(f'{c_state[j]:064b}' for j in range(c_state_size.value)), 2)
    
    @property
    def amplitudes(self):
        if not self.available:
            exec_quantum()

        c_amplitudes = c_void_p()
        c_size       = c_size_t()
        ket_error_warpper(
            self.ket_dump_amplitudes(self, c_amplitudes, c_size)
        )
        for i in range(c_size.value):
            c_real = c_double()
            c_imag = c_double()
            ket_error_warpper(
                self.ket_dump_amp_at(c_amplitudes, c_real, c_imag, i)
            )
            yield c_real.value+c_imag.value*1j

    def show(self, format : Optional[str] = None) -> str:
        r"""Return the quantum state as a string
        
        Use the format starting to change the print format of the basis states:

        * ``i``: print the state in the decimal base
        * ``b``: print the state in the binary base (default)
        * ``i|b<n>``: separate the ``n`` first qubits, the remaining print in the binary base
        * ``i|b<n1>:i|b<n2>[:i|b<n3>...]``: separate the ``n1, n2, n3, ...`` first qubits
        
        :Example:
            
        .. code-block:: ket

            q = quant(19)
            X(ctrl(H(q[0]), X, q[1:])[1::2])
            d = dump(q)

            print(d.show('i'))
            #|87381⟩         (50.00%)
            # 0.707107               ≅      1/√2
            #|436906⟩        (50.00%)
            # 0.707107               ≅      1/√2

            print(d.show('b'))
            #|0010101010101010101⟩   (50.00%)
            # 0.707107               ≅      1/√2
            #|1101010101010101010⟩   (50.00%)
            # 0.707107               ≅      1/√2
            
            print(d.show('i4'))
            #|2⟩|101010101010101⟩    (50.00%)
            # 0.707107               ≅      1/√2
            #|13⟩|010101010101010⟩   (50.00%)
            # 0.707107               ≅      1/√2

            print(d.show('b5:i4'))
            #|00101⟩|5⟩|0101010101⟩  (50.00%)
            # 0.707107               ≅      1/√2
            #|11010⟩|10⟩|1010101010⟩ (50.00%)
            # 0.707107               ≅      1/√2

        Args:
            format: Format string that matchs ``(i|b)\d*(:(i|b)\d+)*``.
        """
        
        dump_str = ''

        if format is not None:
            if format == 'b' or format == 'i':
                format += str(self.size)
            fmt = []
            count = 0
            for b, size in map(lambda f : (f[0], int(f[1:])), format.split(':')):
                fmt.append((b, count, count+size))
                count += size
            if count < self.size:
                fmt.append(('b', count, self.size))
        else:
            fmt = [('b', 0, self.size)]

        fmt_ket = lambda state, begin, end, f : f'|{state[begin:end]}⟩' if f == 'b' else f'|{int(state[begin:end], base=2)}⟩'

        for state, amp in sorted(zip(self.states, self.amplitudes)):
            dump_str += ''.join(fmt_ket(f'{state:0{self.size}b}', b, e, f) for f, b, e in fmt)
            dump_str += f"\t({100*abs(amp)**2:.2f}%)\n"
            real = abs(amp.real) > 1e-10
            real_l0 = amp.real < 0

            imag = abs(amp.imag) > 1e-10
            imag_l0 = amp.imag < 0

            sqrt_dem = 1/abs(amp)**2
            use_sqrt = abs(round(sqrt_dem)-sqrt_dem) < .001
            sqrt_dem = f'/√{round(1/abs(amp)**2)}'

            if real and imag:
                sqrt_dem = f'/√{round(2*(1/abs(amp)**2))}'
                sqrt_num = ('(-1' if real_l0 else ' (1')+('-i' if imag_l0 else '+i')
                sqrt_str = f'\t≅ {sqrt_num}){sqrt_dem}\n' if use_sqrt else '\n'
                dump_str += f"{amp.real:9.6f}{amp.imag:+.6f}i"+sqrt_str 
            elif real:
                sqrt_num = '  -1' if real_l0 else '   1'
                sqrt_str = f'\t≅   {sqrt_num}{sqrt_dem}\n' if use_sqrt else '\n'
                dump_str += f"{amp.real:9.6f}       "+sqrt_str
            else:
                sqrt_num = '  -i' if imag_l0 else '   i'
                sqrt_str = f'\t≅   {sqrt_num}{sqrt_dem}\n' if use_sqrt else '\n'
                dump_str += f" {amp.imag:17.6f}i"+sqrt_str

        return dump_str
            
    @property
    def available(self):
        c_value = c_bool()
        ket_error_warpper(
            self.ket_dump_available(self, c_value)
        )
        return c_value.value

    @property
    def index(self):
        c_value = c_uint()
        ket_error_warpper(
            self.ket_dump_index(self, c_value)
        )
        return c_value.value

    @property
    def process_id(self):
        c_value = c_uint()
        ket_error_warpper(
            self.ket_dump_process_id(self, c_value)
        )
        return c_value.value
    
    def __repr__(self):
        return f"<Ket 'dump' {(self.process_id, self.index)}>"

class future:
    ket_future_new = libketc.ket_future_new
    ket_future_new.argtypes = [POINTER(c_void_p)]

    ket_future_delete = libketc.ket_future_delete
    ket_future_delete.argtypes = [c_void_p]

    ket_future_value = libketc.ket_future_value
    ket_future_value.argtypes = [c_void_p, POINTER(c_long)]

    ket_future_set = libketc.ket_future_set
    ket_future_set = [c_void_p, c_void_p]

    ket_future_available = libketc.ket_future_available
    ket_future_available.argtypes = [c_void_p, POINTER(c_bool)]

    ket_future_index = libketc.ket_future_index
    ket_future_index.argtypes = [c_void_p, POINTER(c_uint)]

    ket_future_process_id = libketc.ket_future_process_id
    ket_future_process_id.argtypes = [c_void_p, POINTER(c_uint)]

    ket_future_op = libketc.ket_future_op

    def __init__(self, value = None):
        self._as_parameter_ = c_void_p()
        ket_error_warpper(
            self.ket_future_new(byref(self._as_parameter_))
        )

    def __del__(self):
        ket_error_warpper(
            self.ket_future_delete(self)
        )

    def __getattr__(self, name):
        if name != "value":
            raise AttributeError(name)
    
        if not self.available:
            exec_quantum()

        value = c_long()
        ket_error_warpper(
            self.ket_future_value(self, value)
        )
        return value.value
    
    def __setattr__(self, name, value):
        if name == "value":
            ket_error_warpper(
                self.ket_future_set(self, value)
            )
        else:
            super().__setattr__(name, value)

    @property
    def available(self):
        c_value = c_bool()
        ket_error_warpper(
            self.ket_future_available(self, c_value)
        )
        return c_value.value

    @property
    def index(self):
        c_value = c_uint()
        ket_error_warpper(
            self.ket_future_index(self, c_value)
        )
        return c_value.value

    @property
    def process_id(self):
        c_value = c_uint()
        ket_error_warpper(
            self.ket_future_process_id(self, c_value)
        )
        return c_value.value

    def __add__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_ADD, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_ADD, KET_INT_FI, self, int(other))
            )
        return result

    def __sub__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_ADD, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_ADD, KET_INT_FI, self, int(other))
            )
        return result

    def __mul__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_MUL, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_MUL, KET_INT_FI, self, int(other))
            )
        return result

    def __truediv__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_DIV, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_DIV, KET_INT_FI, self, int(other))
            )
        return result

    def __floordiv__(self, other):
        return self.__truediv__(other)

    def __lshift__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_SLL, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_SLL, KET_INT_FI, self, int(other))
            )
        return result

    def __rshift__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_SRL, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_SRL, KET_INT_FI, self, int(other))
            )
        return result

    def __and__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_AND, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_AND, KET_INT_FI, self, int(other))
            )
        return result

    def __xor__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_XOR, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_XOR, KET_INT_FI, self, int(other))
            )
        return result

    def __or__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_OR, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_OR, KET_INT_FI, self, int(other))
            )
        return result

    def __radd__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_ADD, KET_INT_IF, int(other), self)
        )
        return result

    def __rsub__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_SUB, KET_INT_IF, int(other), self)
        )
        return result

    def __rmul__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_MUL, KET_INT_IF, int(other), self)
        )
        return result

    def __rtruediv__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_DIV, KET_INT_IF, int(other), self)
        )
        return result

    def __rfloordiv__(self, other):
        return self.__rtruediv__(other)

    def __rlshift__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_SLL, KET_INT_IF, int(other), self)
        )
        return result

    def __rrshift__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_SRL, KET_INT_IF, int(other), self)
        )
        return result

    def __rand__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_AND, KET_INT_IF, int(other), self)
        )
        return result

    def __rxor__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_XOR, KET_INT_IF, int(other), self)
        )
        return result

    def __ror__(self, other):
        self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_long, c_void_p]
        result = future()
        ket_error_warpper(
            self.ket_future_op(result, KET_INT_OR, KET_INT_IF, int(other), self)
        )
        return result

    def __lt__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_LT, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_LT, KET_INT_FI, self, int(other))
            )
        return result

    def __le__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_LEQ, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_LEQ, KET_INT_FI, self, int(other))
            )
        return result

    def __eq__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_EQ, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_EQ, KET_INT_FI, self, int(other))
            )
        return result

    def __ne__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_NEQ, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_NEQ, KET_INT_FI, self, int(other))
            )
        return result

    def __gt__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_GT, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_GT, KET_INT_FI, self, int(other))
            )
        return result

    def __ge__(self, other):
        result = future()
        if isinstance(other, future):
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_void_p]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_GEQ, KET_INT_FF, self, other)
            )
        else:
            self.ket_future_op.argtypes = [c_void_p, c_int, c_int, c_void_p, c_long]
            ket_error_warpper(
                self.ket_future_op(result, KET_INT_GEQ, KET_INT_FI, self, int(other))
            )
        return result

    def __repr__(self):
        return f"<Ket 'future' {(self.process_id, self.index)}>"

class label:
    ket_label_new = libketc.ket_label_new
    ket_label_new.argtypes = [POINTER(c_void_p)]

    ket_label_delete = libketc.ket_label_delete
    ket_label_delete.argtypes = [c_void_p]

    ket_label_index = libketc.ket_label_index
    ket_label_index.argtypes = [c_void_p, POINTER(c_uint)]

    ket_label_process_id = libketc.ket_label_process_id
    ket_label_process_id.argtypes = [c_void_p, POINTER(c_uint)]

    def __init__(self):
        self._as_parameter_ = c_void_p()
        ket_error_warpper(
            self.ket_label_new(byref(self._as_parameter_))
        )
        process_top().get_label(self)

    def __del__(self):
        ket_error_warpper(
            self.ket_label_delete(self)
        )

    def begin(self):
        process_top().open_block(self)

    @property
    def index(self):
        c_value = c_uint()
        ket_error_warpper(
            self.ket_label_index(self, c_value)
        )
        return c_value.value

    @property
    def process_id(self):
        c_value = c_uint()
        ket_error_warpper(
            self.ket_label_process_id(self, c_value)
        )
        return c_value.value

    def __repr__(self):
        return f"<Ket 'label' {(self.process_id, self.index)}>"

class process:
    ket_process_new = libketc.ket_process_new
    ket_process_new.argtypes = [POINTER(c_void_p), c_uint]

    ket_process_delete = libketc.ket_process_delete
    ket_process_delete.argtypes = [c_void_p]

    ket_process_alloc = libketc.ket_process_alloc
    ket_process_alloc.argtypes = [c_void_p, c_void_p, c_bool]

    ket_process_free = libketc.ket_process_alloc
    ket_process_free.argtypes = [c_void_p, c_void_p, c_bool]

    ket_process_gate = libketc.ket_process_gate
    ket_process_gate.argtypes = [c_void_p, c_int, c_void_p, c_double]

    ket_process_measure = libketc.ket_process_measure

    ket_process_ctrl_push = libketc.ket_process_ctrl_push

    ket_process_ctrl_pop = libketc.ket_process_ctrl_pop
    ket_process_ctrl_pop.argtypes = [c_void_p]

    ket_process_adj_begin = libketc.ket_process_adj_begin
    ket_process_adj_begin.argtypes = [c_void_p]

    ket_process_adj_end = libketc.ket_process_adj_end
    ket_process_adj_end.argtypes = [c_void_p]

    ket_process_get_label = libketc.ket_process_get_label
    ket_process_get_label.argtypes = [c_void_p, c_void_p]

    ket_process_open_block = libketc.ket_process_open_block
    ket_process_open_block.argtypes = [c_void_p, c_void_p]

    ket_process_jump = libketc.ket_process_jump
    ket_process_jump.argtypes = [c_void_p, c_void_p]

    ket_process_branch = libketc.ket_process_branch
    ket_process_branch.argtypes = [c_void_p, c_void_p, c_void_p]

    ket_process_dump = libketc.ket_process_dump

    ket_process_run = libketc.ket_process_run
    ket_process_run.argtypes = [c_void_p]

    ket_process_exec_time = libketc.ket_process_exec_time
    ket_process_exec_time.argtypes = [c_void_p, POINTER(c_double)]

    ket_process_id = libketc.ket_process_id
    ket_process_id.argtypes = [c_void_p, POINTER(c_uint)]

    def __init__(self, process_id : int):
        self._as_parameter_ = c_void_p()
        ket_error_warpper(
            self.ket_process_new(byref(self._as_parameter_), process_id)
        )

    def __del__(self):
        ket_error_warpper(
            self.ket_process_delete(self)
        )
    
    def alloc(self, dirty = False):
        q = qubit()
        ket_error_warpper(
            self.ket_process_alloc(self, q, dirty)
        )
        return q    

    def free(self, qubit : qubit, dirty = False):
        ket_error_warpper(
            self.ket_process_alloc(self, qubit, dirty)
        )

    def gate(self, gate : int, qubit : qubit, param : float = 0):
        ket_error_warpper(
            self.ket_process_gate(self, gate, qubit, param)
        )
    
    def measure(self, *qubits):
        self.ket_process_measure.argtypes = [c_void_p, c_void_p, c_int] + [c_void_p for _ in range(len(qubits))]
        result = future()
        ket_error_warpper(
            self.ket_process_measure(self, result, len(qubits), *qubits)
        )
        return result

    def ctrl_push(self, *qubits):
        self.ket_process_ctrl_push.argtypes = [c_void_p, c_int] + [c_void_p for _ in range(len(qubits))]
        ket_error_warpper(
            self.ket_process_ctrl_push(self, len(qubits), *qubits)
        )

    def ctrl_pop(self):
        ket_error_warpper(
            self.ket_process_ctrl_pop(self)
        )

    def adj_begin(self):
        ket_error_warpper(
            self.ket_process_adj_begin(self)
        )

    def adj_end(self):
        ket_error_warpper(
            self.ket_process_adj_end(self)
        )

    def get_label(self, label_var):
        ket_error_warpper(
            self.ket_process_get_label(self, label_var)
        )

    def open_block(self, label : label):
        ket_error_warpper(
            self.ket_process_open_block(self, label)
        )

    def jump(self, label : label):
        ket_error_warpper(
            self.ket_process_jump(self, label)
        )

    def branch(self, test : future, then : label, otherwise : label):
        ket_error_warpper(
            self.ket_process_branch(self, test, then, otherwise)
        )

    def run(self):
        ket_error_warpper(
            self.ket_process_run(self)
        )

    @property
    def exec_time(self):
        value = c_double()
        ket_error_warpper(
            self.ket_process_exec_time(self, value)
        )
        return value.value

    @property
    def id(self):
        value = c_uint()
        ket_error_warpper(
            self.ket_process_id(self, value)
        )
        return value.value    
    
    def dump(self, dump_var, *qubits):
        self.ket_process_dump.argtypes = [c_void_p, c_void_p, c_int] + [c_void_p for _ in range(len(qubits))]
        ket_error_warpper(
            self.ket_process_dump(self, dump_var, len(qubits), *qubits)
        )

    def __repr__(self):
        return f"<Ket 'process' {(self.id)}>"

process_count = 1
process_stack = [process(0)]

def process_begin():
    global process_count
    global process_stack
    process_stack.append(process(process_count))
    process_count += 1

def process_end():
    global process_stack
    return process_stack.pop()

def process_top():
    global process_stack
    return process_stack[-1]

def exec_quantum():
    environ["KQE_SEED"] = str(randint(0, 1 << 31))
    process_end().run()
    process_begin()

def measure(*qubits):
    return process_top().measure(*qubits)

def ctrl_push(*qubits):
    return process_top().ctrl_push(*qubits)

def ctrl_pop():
    return process_top().ctrl_pop()

def adj_begin():
    return process_top().adj_begin()

def adj_end():
    return process_top().adj_end()

def jump(label):
    return process_top().jump(label)

def branch(test, then, otherwise):
    return process_top().branch(test, then, otherwise)

def X(q):
    for qubit in q.qubits:
        process_top().gate(KET_PAULI_X, qubit, 0.0)

def Y(q):
    for qubit in q.qubits:
        process_top().gate(KET_PAULI_Y, qubit, 0.0)

def Z(q):
    for qubit in q.qubits:
        process_top().gate(KET_PAULI_Z, qubit, 0.0)

def H(q):
    for qubit in q.qubits:
        process_top().gate(KET_HADAMARD, qubit, 0.0)

def S(q):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, pi/2)

def SD(q):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, -pi/2)

def T(q):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, pi/4)

def TD(q):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, -pi/4)

def phase(lambda_, q):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, lambda_)

def RX(theta, q):
    for qubit in q.qubits:
        process_top().gate(KET_ROTATION_X, qubit, theta)

def RY(theta, q):
    for qubit in q.qubits:
        process_top().gate(KET_ROTATION_Y, qubit, theta)

def RZ(theta, q):
    for qubit in q.qubits:
        process_top().gate(KET_ROTATION_Z, qubit, theta)
    