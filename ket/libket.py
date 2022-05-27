from __future__ import annotations
# Licensed under the Apache License, Version 2.0;
# Copyright 2022 Evandro Chagas Ribeiro da Rosa
from math import pi
from ctypes import *
from os import environ
from os.path import dirname

__all__ = ['quant', 'future', 'dump', 'qc_int', 'exec_quantum']


def load_libket():
    if "LIBKET_PATH" in environ:
        libket_path = environ["LIBKET_PATH"]
    else:
        libket_path = dirname(__file__)+"/lib/libket.so"

    return cdll.LoadLibrary(libket_path)


class libket_error(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


KET_SUCCESS = 0
KET_ERROR = 1

KET_PAULI_X = 0
KET_PAULI_Y = 1
KET_PAULI_Z = 2
KET_HADAMARD = 3
KET_PHASE = 4
KET_ROTATION_X = 5
KET_ROTATION_Y = 6
KET_ROTATION_Z = 7

KET_INT_EQ = 0
KET_INT_NEQ = 1
KET_INT_GT = 2
KET_INT_GEQ = 3
KET_INT_LT = 4
KET_INT_LEQ = 5
KET_INT_ADD = 6
KET_INT_SUB = 7
KET_INT_MUL = 8
KET_INT_DIV = 9
KET_INT_SLL = 10
KET_INT_SRL = 11
KET_INT_AND = 12
KET_INT_OR = 13
KET_INT_XOR = 14

libket = load_libket()
ket_error_message = libket.ket_error_message
ket_error_message.argtypes = [POINTER(c_size_t)]
ket_error_message.restype = POINTER(c_ubyte)


def ket_error_warpper(error: c_int):
    if error == KET_ERROR:
        size = c_size_t()
        error_msg = ket_error_message(size)
        raise libket_error(bytearray(error_msg[:size.value]).decode())


def ket_error_warpper_addr(addr: c_void_p):
    if addr == None:
        size = c_size_t()
        error_msg = ket_error_message(size)
        raise libket_error(bytearray(error_msg[:size.value]).decode())
    else:
        return addr


class qubit:
    """Qubit reference

        Intended for internal use.
    """

    ket_qubit_delete = libket.ket_qubit_delete
    ket_qubit_delete.argtypes = [c_void_p]
    ket_qubit_delete.restype = None

    ket_qubit_index = libket.ket_qubit_index
    ket_qubit_index.argtypes = [c_void_p]
    ket_qubit_index.restype = c_uint32

    ket_qubit_pid = libket.ket_qubit_pid
    ket_qubit_pid.argtypes = [c_void_p]
    ket_qubit_pid.restype = c_uint32

    ket_qubit_allocated = libket.ket_qubit_allocated
    ket_qubit_allocated.argtypes = [c_void_p]
    ket_qubit_allocated.restype = c_bool

    ket_qubit_measured = libket.ket_qubit_measured
    ket_qubit_measured.argtypes = [c_void_p]
    ket_qubit_measured.restype = c_bool

    def __init__(self, ptr: c_void_p):
        self._as_parameter_ = ptr

    def __del__(self):
        self.ket_qubit_delete(self)

    @property
    def index(self) -> int:
        return self.ket_qubit_pid(self)

    @property
    def pid(self) -> int:
        return self.ket_qubit_pid(self)

    @property
    def allocated(self) -> bool:
        return self.ket_qubit_allocated(self).value

    @property
    def measured(self) -> bool:
        return self.ket_qubit_measured(self).value

    def __repr__(self) -> str:
        return f"<Ket 'qubit' {(self.process_id, self.index)}>"


class quant:
    r"""Create list of qubits

    Allocate ``size`` qubits in the state :math:`\left|0\right>`.

    If ``dirty`` is ``True``, allocate ``size`` qubits in an unknown state.

    warning:
        Using dirty qubits may have side effects due to previous entanglements.

    Qubits allocated using the ``with`` statement must be free at the end of the scope.

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
    two :class:`~ket.libket.quant`.

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

    Args:
        size: The number of qubits to allocate.
        dirty: If ``True``, allocate ``size`` qubits at an unknown state.
        qubits: Initialize the qubit list without allocating. Intended for internal use.
    """

    def __init__(self, size: int = 1, dirty: bool = False, *, qubits: list[qubit] | None = None):
        if qubits is not None:
            self.qubits = qubits
        else:
            self.qubits = [process_top().alloc(dirty) for _ in range(size)]

    def __add__(self, other: quant) -> quant:
        return quant(qubits=self.qubits+other.qubits)

    def at(self, index: [int]) -> quant:
        r"""Return qubits at ``index``

        Create a new :class:`~ket.libket.quant` with the qubit references at the
        position defined by the ``index`` list.

        :Example:

        .. code-block:: ket

            q = quant(20)        
            odd = q.at(range(1, len(q), 2)) # = q[1::2]

        Args:
            index: List of indexes.
        """

        return quant(qubits=[self.qubits[i] for i in index])

    def free(self, dirty: bool = False):
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
            qubits_str = ', '.join(
                str(self.qubits[i].index) for i in range(5))+', ...'
        else:
            qubits_str = ', '.join(str(q.index) for q in self.qubits)

        return f"<Ket 'quant' ({qubits_str})>"


class dump:
    """Create a snapshot with the current quantum state of ``qubits``.

    Gathering any information from a :class:`~ket.libket.dump` triggers the quantum execution.

    :Example:

    .. code-block:: ket

        a, b = quant(2)
        with around(cnot(H, I), a, b):
            Y(a)
            inside = dump(a+b)
        outside = dump(a+b)

        print('inside:')
        print(inside.show())
        #inside:
        #|01⟩    (50.00%)
        #         -0.707107i     ≅     -i/√2
        #|10⟩    (50.00%)
        #          0.707107i     ≅      i/√2
        print('outside:')
        print(outside.show())
        #outside:
        #|11⟩    (100.00%)
        #         -1.000000i     ≅     -i/√1

    :param qubits: Qubits to dump.
    """

    ket_dump_delete = libket.ket_dump_delete
    ket_dump_delete.argtypes = [c_void_p]
    ket_dump_delete.restype = None

    ket_dump_states = libket.ket_dump_states
    ket_dump_states.argtypes = [c_void_p, POINTER(c_size_t)]
    ket_dump_states.restype = c_void_p

    ket_dump_get_state = libket.ket_dump_get_state
    ket_dump_get_state.argtypes = [c_void_p, c_size_t, POINTER(c_size_t)]
    ket_dump_get_state.restype = POINTER(c_uint64)

    ket_dump_amplitudes_real = libket.ket_dump_amplitudes_real
    ket_dump_amplitudes_real.argtypes = [c_void_p, POINTER(c_size_t)]
    ket_dump_amplitudes_real.restype = POINTER(c_double)

    ket_dump_amplitudes_img = libket.ket_dump_amplitudes_img
    ket_dump_amplitudes_img.argtypes = [c_void_p, POINTER(c_size_t)]
    ket_dump_amplitudes_img.restype = POINTER(c_double)

    ket_dump_available = libket.ket_dump_available
    ket_dump_available.argtypes = [c_void_p]
    ket_dump_available.restype = c_bool

    def __init__(self, qubits: quant):
        self._as_parameter_ = process_top().dump(*qubits.qubits)
        self.size = len(qubits.qubits)

    def __del__(self):
        self.ket_dump_delete(self)

    @property
    def states(self) -> [int]:
        """List of basis states"""

        if not self.available:
            exec_quantum()

        size = c_size_t()
        states = self.ket_dump_states(self, size)
        size = size.value

        for i in range(size):
            state_size = c_size_t()
            state = self.ket_dump_get_state(states, i, state_size)
            state_size = state_size.value

            yield int(''.join(f'{state[j]:064b}' for j in range(state_size)), 2)

    @property
    def amplitudes(self) -> [complex]:
        """List of probability amplitudes"""

        if not self.available:
            exec_quantum()

        size = c_size_t()
        real = self.ket_dump_amplitudes_real(self, size)
        _size = c_size_t()
        img = self.ket_dump_amplitudes_img(self, _size)
        size = size.value
        assert(size == _size.value)

        for i in range(size):
            yield real[i]+img[i]*1j

    @property
    def probability(self) -> [float]:
        for amp in self.amplitudes:
            yield abs(amp)**2

    def show(self, format: str | None = None) -> str:
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
            #|87381⟩ (50.00%)
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

        if format is not None:
            if format == 'b' or format == 'i':
                format += str(self.size)
            fmt = []
            count = 0
            for b, size in map(lambda f: (f[0], int(f[1:])), format.split(':')):
                fmt.append((b, count, count+size))
                count += size
            if count < self.size:
                fmt.append(('b', count, self.size))
        else:
            fmt = [('b', 0, self.size)]

        def fmt_ket(state, begin, end,
                    f): return f'|{state[begin:end]}⟩' if f == 'b' else f'|{int(state[begin:end], base=2)}⟩'

        def state_amp_str(state, amp):
            dump_str = ''.join(
                fmt_ket(f'{state:0{self.size}b}', b, e, f) for f, b, e in fmt)
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
                sqrt_num = ('(-1' if real_l0 else ' (1') + \
                    ('-i' if imag_l0 else '+i')
                sqrt_str = f'\t≅ {sqrt_num}){sqrt_dem}' if use_sqrt else ''
                dump_str += f"{amp.real:9.6f}{amp.imag:+.6f}i"+sqrt_str
            elif real:
                sqrt_num = '  -1' if real_l0 else '   1'
                sqrt_str = f'\t≅   {sqrt_num}{sqrt_dem}' if use_sqrt else ''
                dump_str += f"{amp.real:9.6f}       "+sqrt_str
            else:
                sqrt_num = '  -i' if imag_l0 else '   i'
                sqrt_str = f'\t≅   {sqrt_num}{sqrt_dem}' if use_sqrt else ''
                dump_str += f" {amp.imag:17.6f}i"+sqrt_str

            return dump_str

        return '\n'.join(state_amp_str(state, amp) for state, amp in sorted(zip(self.states, self.amplitudes), key=lambda k: k[0]))

    @property
    def expected_values(self):
        """X, Y, and Z expected values for one qubit"""

        if self.size != 1:
            raise RuntimeError(
                'Cannot calculate X, Y, and Z expected values from a dump with more than 1 qubit')

        def exp_x(alpha, beta): return (
            beta.conjugate()*alpha+alpha.conjugate()*beta).real
        def exp_y(alpha, beta): return (1j*beta.conjugate()
                                        * alpha-1j*alpha.conjugate()*beta).real

        def exp_z(alpha, beta): return pow(abs(alpha), 2)-pow(abs(beta), 2)
        alpha = 0
        beta = 0
        for a, s in zip(self.amplitudes, self.states):
            if s == 0:
                alpha = a
            else:
                beta = a
        return [exp_x(alpha, beta), exp_y(alpha, beta), exp_z(alpha, beta)]

    def sphere(self):
        """Result a Bloch sphere

        QuTiP and Matplotlib are needed to generate and plot the sphere.
        """
        try:
            import qutip
        except ImportError as e:
            from sys import stderr
            print("Unable to import QuTiP, try installing:", file=stderr)
            print("\tpip install qutip", file=stderr)
            raise e

        b = qutip.Bloch()
        b.add_vectors(self.expected_values)
        return b

    @property
    def available(self) -> bool:
        return self.ket_dump_available(self)

    def __repr__(self) -> str:
        return "<Ket 'dump'>"


class future:
    """64-bits integer on the quantum computer

    Store a reference to a 64-bits integer available in the quantum computer. 

    The integer value are available to the classical computer only after the
    quantum execution.

    The follwing binary operations are available between
    :class:`~ket.libket.future` variables and ``int``: 

        ``==``, ``!=``, ``<``, ``<=``,
        ``>``, ``>=``, ``+``, ``-``, ``*``, ``/``, ``<<``, ``>>``, ``and``,
        ``xor``, and ``or``.

    A new :class:`~ket.libket.future` variable is created with a quantum
    :func:`~ket.standard.measure` (1) , binary operation with a
    :class:`~ket.libket.future` (2), or directly initialization with a ``int``
    (2).

    .. code-block:: ket

        q = H(quant(2))
        a = measure(q) # 1
        b = a*3        # 2
        c = qc_int(42) # 3



    Writing to the attribute ``value`` of a :class:`~ket.libket.future` variable
    passes the information to the quantum computer. 
    Reading the attribute ``value`` triggers the quantum execution. 

    If the test expression of an ``if-then-else`` or ``while`` is type future,
    Ket passes the statement to the quantum computer.

    :Example:

    .. code-block:: ket

        q = quant(2)
        with quant() as aux:
            # Create variable done on the quantum computer
            done = qc_int(False) 
            while done != True:
                H(q)
                ctrl(q, X, aux)
                res = measure(aux)
                if res == 0:
                    # Update variable done on the quantum computer
                    done.value = True
                else:
                    X(q+aux)
            aux.free()
        # Get the measurement from the quantum computer
        # triggering the quantum execution
        result = measure(q).value 

    """

    ket_future_delete = libket.ket_future_delete
    ket_future_delete.argtypes = [c_void_p]
    ket_future_delete.restype = None

    ket_future_value = libket.ket_future_value
    ket_future_value.argtypes = [c_void_p, POINTER(c_bool)]
    ket_future_value.restype = c_int64

    ket_future_index = libket.ket_future_index
    ket_future_index.argtypes = [c_void_p]
    ket_future_index.restype = c_uint32

    ket_future_pid = libket.ket_future_pid
    ket_future_pid.argtypes = [c_void_p]
    ket_future_pid.restype = c_uint32

    def __init__(self, ptr):
        self._as_parameter_ = ptr
        self._value = None

    def __del__(self):
        self.ket_future_delete(self)

    def __getattr__(self, name):
        if name == "value":
            if not self.available:
                exec_quantum()
                self.available
            return self._value
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "value":
            if not isinstance(value, future):
                value = qc_int(value)
            process_top().int_set(self, value)
        else:
            super().__setattr__(name, value)

    @property
    def available(self) -> bool:
        if self._value is None:
            available = c_bool()
            value = self.ket_future_value(self, available)
            if available.value:
                self._value = value
            return available.value
        else:
            return True

    @property
    def index(self) -> int:
        return self.ket_future_pid(self)

    @property
    def pid(self) -> int:
        return self.ket_future_pid(self)

    def __add__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_ADD, self, other)

    def __sub__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_SUB, self, other)

    def __mul__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_MUL, self, other)

    def __truediv__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_DIV, self, other)

    def __floordiv__(self, other: future | int) -> future:
        return self.__truediv__(other)

    def __lshift__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_SLL, self, other)

    def __rshift__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_SRL, self, other)

    def __and__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_AND, self, other)

    def __xor__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_XOR, self, other)

    def __or__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_OR, self, other)

    def __radd__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_ADD, other, self)

    def __rsub__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_SUB, other, self)

    def __rmul__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_MUL, other, self)

    def __rtruediv__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_DIV, other, self)

    def __rfloordiv__(self, other: future | int) -> future:
        return self.__rtruediv__(other)

    def __rlshift__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_SLL, other, self)

    def __rrshift__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_SRL, other, self)

    def __rand__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_AND, other, self)

    def __rxor__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_XOR, other, self)

    def __ror__(self, other: future | int) -> future:
        other = qc_int(other)
        return process_top().int_op(KET_INT_OR, other, self)

    def __lt__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_LT, self, other)

    def __le__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_LEQ, self, other)

    def __eq__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_EQ, self, other)

    def __ne__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_NEQ, self, other)

    def __gt__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_GT, self, other)

    def __ge__(self, other: future | int) -> future:
        if not isinstance(other, future):
            other = qc_int(other)
        return process_top().int_op(KET_INT_GEQ, self, other)

    def __repr__(self):
        return f"<Ket 'future' {(self.pid, self.index)}>"


class label:

    ket_label_delete = libket.ket_label_delete
    ket_label_delete.argtypes = [c_void_p]
    ket_label_delete.restype = None

    ket_label_index = libket.ket_label_index
    ket_label_index.argtypes = [c_void_p]
    ket_label_index.restype = c_uint32

    ket_label_pid = libket.ket_label_pid
    ket_label_pid.argtypes = [c_void_p]
    ket_label_pid.restype = c_uint32

    def __init__(self):
        self._as_parameter_ = process_top().get_label()

    def __del__(self):
        self.ket_label_delete(self)

    def begin(self):
        process_top().open_block(self)

    @property
    def index(self):
        return self.ket_label_pid(self)

    @property
    def pid(self):
        return self.ket_label_pid(self)

    def __repr__(self):
        return f"<Ket 'label' {(self.pid, self.index)}>"


class process:
    ket_process_new = libket.ket_process_new
    ket_process_new.argtypes = [c_uint32]
    ket_process_new.restype = c_void_p

    ket_process_delete = libket.ket_process_delete
    ket_process_delete.argtypes = [c_void_p]
    ket_process_delete.restype = None

    ket_process_alloc = libket.ket_process_alloc
    ket_process_alloc.argtypes = [c_void_p, c_bool]
    ket_process_alloc.restype = c_void_p

    ket_process_free = libket.ket_process_free
    ket_process_free.argtypes = [c_void_p, c_void_p, c_bool]

    ket_process_apply_gate = libket.ket_process_apply_gate
    ket_process_apply_gate.argtypes = [c_void_p, c_uint32, c_double, c_void_p]

    ket_process_measure = libket.ket_process_measure
    ket_process_measure.argtypes = [c_void_p, c_void_p, c_size_t]
    ket_process_measure.restype = c_void_p

    ket_process_ctrl_push = libket.ket_process_ctrl_push
    ket_process_ctrl_push.argtypes = [c_void_p, c_void_p, c_size_t]

    ket_process_ctrl_pop = libket.ket_process_ctrl_pop
    ket_process_ctrl_pop.argtypes = [c_void_p]

    ket_process_adj_begin = libket.ket_process_adj_begin
    ket_process_adj_begin.argtypes = [c_void_p]

    ket_process_adj_end = libket.ket_process_adj_end
    ket_process_adj_end.argtypes = [c_void_p]

    ket_process_get_label = libket.ket_process_get_label
    ket_process_get_label.argtypes = [c_void_p]
    ket_process_get_label.restype = c_void_p

    ket_process_open_block = libket.ket_process_open_block
    ket_process_open_block.argtypes = [c_void_p, c_void_p]

    ket_process_jump = libket.ket_process_jump
    ket_process_jump.argtypes = [c_void_p, c_void_p]

    ket_process_branch = libket.ket_process_branch
    ket_process_branch.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]

    ket_process_dump = libket.ket_process_dump
    ket_process_dump.argtypes = [c_void_p, c_void_p, c_size_t]
    ket_process_dump.restype = c_void_p

    ket_process_add_int_op = libket.ket_process_add_int_op
    ket_process_add_int_op.argtypes = [c_void_p, c_uint32, c_void_p, c_void_p]
    ket_process_add_int_op.restype = c_void_p

    ket_process_int_new = libket.ket_process_int_new
    ket_process_int_new.argtypes = [c_void_p, c_int64]
    ket_process_int_new.restype = c_void_p

    ket_process_int_set = libket.ket_process_int_set
    ket_process_int_set.argtypes = [c_void_p, c_void_p, c_void_p]

    ket_process_exec_time = libket.ket_process_exec_time
    ket_process_exec_time.argtypes = [c_void_p, POINTER(c_bool)]
    ket_process_exec_time.restype = c_double

    ket_process_get_quantum_code_as_json = libket.ket_process_get_quantum_code_as_json
    ket_process_get_quantum_code_as_json.argtypes = [
        c_void_p, POINTER(c_size_t)]
    ket_process_get_quantum_code_as_json.restype = POINTER(c_ubyte)

    ket_process_get_quantum_code_as_bin = libket.ket_process_get_quantum_code_as_bin
    ket_process_get_quantum_code_as_bin.argtypes = [
        c_void_p, POINTER(c_size_t)]
    ket_process_get_quantum_code_as_bin.restype = POINTER(c_ubyte)

    def __init__(self, pid: int):
        self.pid = pid
        self._as_parameter_ = ket_error_warpper_addr(
            self.ket_process_new(pid))

    def __del__(self):
        self.ket_process_delete(self)

    def alloc(self, dirty=False):
        ptr = ket_error_warpper_addr(self.ket_process_alloc(self, dirty))
        return qubit(ptr)

    def free(self, qubit: qubit, dirty=False):
        ket_error_warpper(self.ket_process_free(self, qubit, dirty))

    def gate(self, gate: int, qubit: qubit, param: float = 0):
        ket_error_warpper(self.ket_process_apply_gate(
            self, gate, param, qubit))

    def measure(self, *qubits):
        qubits = (c_void_p*len(qubits))(*(q._as_parameter_ for q in qubits))
        ptr = ket_error_warpper_addr(
            self.ket_process_measure(self, qubits, len(qubits)))
        return future(ptr)

    def new_int(self, value):
        ptr = ket_error_warpper_addr(self.ket_process_int_new(self, value))
        return future(ptr)

    # def plugin(self, name, args, *qubits):
    #    self.ket_process_plugin.argtypes = [
    #        c_void_p, c_char_p, c_char_p, c_int] + [c_void_p for _ in range(len(qubits))]
    #    ket_error_warpper(
    #        self.ket_process_plugin(
    #            self, name.encode(), args.encode(), len(qubits), *qubits)
    #    )

    def ctrl_push(self, *qubits):
        qubits = (c_void_p*len(qubits))(*(q._as_parameter_ for q in qubits))
        ket_error_warpper(self.ket_process_ctrl_push(
            self, qubits, len(qubits)))

    def ctrl_pop(self):
        ket_error_warpper(self.ket_process_ctrl_pop(self))

    def adj_begin(self):
        ket_error_warpper(self.ket_process_adj_begin(self))

    def adj_end(self):
        ket_error_warpper(self.ket_process_adj_end(self))

    def get_label(self):
        return ket_error_warpper_addr(self.ket_process_get_label(self))

    def open_block(self, label: label):
        ket_error_warpper(self.ket_process_open_block(self, label))

    def jump(self, label: label):
        ket_error_warpper(self.ket_process_jump(self, label))

    def branch(self, test: future, then: label, otherwise: label):
        ket_error_warpper(self.ket_process_branch(self, test, then, otherwise))

    @property
    def exec_time(self):
        available = c_bool()
        time = self.ket_process_exec_time(self, available)
        if available.value:
            return time.value
        else:
            return None

    # def timeout(self, time):
    #    ket_error_warpper(
    #        self.ket_process_timeout(self, time)
    #    )

    def dump(self, *qubits):
        qubits = (c_void_p*len(qubits))(*(q._as_parameter_ for q in qubits))
        ptr = ket_error_warpper_addr(self.ket_process_dump(
            self, qubits, len(qubits)))
        return ptr

    def int_set(self, result, value):
        ket_error_warpper(self.ket_process_int_set(self, result, value))

    def int_op(self, op, lhs, rhs):
        ptr = ket_error_warpper_addr(
            self.ket_process_add_int_op(self, op, lhs, rhs))
        return future(ptr)

    def get_quantum_code_as_json(self):
        size = c_size_t()
        data = ket_error_warpper_addr(
            self.ket_process_get_quantum_code_as_json(self, size))
        return bytearray(data[:size.value]).decode()

    def get_quantum_code_as_bin(self):
        size = c_size_t()
        data = ket_error_warpper_addr(
            self.ket_process_get_quantum_code_as_bin(self, size))
        return bytearray(data[:size.value])

    def __repr__(self):
        return f"<Ket 'process' {(self.pid)}>"


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


quantum_execution_target = None


def set_quantum_execution_target(func):
    global quantum_execution_target
    quantum_execution_target = func


def exec_quantum():
    """Call the quantum execution"""

    global quantum_execution_target

    error = None
    try:
        quantum_execution_target(process_end())
    except Exception as e:
        error = e
    process_begin()
    if error:
        raise error


def qc_int(value: int) -> future:
    """Instantiate an integer on the quantum computer

    args:
        value: Initial value.
    """

    return process_top().new_int(value)


def measure(q: quant):
    size = len(q)
    if size <= 64:
        return process_top().measure(*q.qubits)
    else:
        return [process_top().measure(*q.qubits[i:min(i+63, size)]) for i in reversed(range(0, size, 63))]


def plugin(name: str, args: str, qubits: quant):
    """Apply plugin

    .. note::

        Plugin availability depends on the quantum execution target.

    args:
        name: Plugin name.
        args: Plugin argument string.
        qubits: Affected qubits.
    """
    process_top().plugin(name, args, *qubits.qubits)


def ctrl_push(q: quant):
    return process_top().ctrl_push(*q.qubits)


def ctrl_pop():
    return process_top().ctrl_pop()


def adj_begin():
    return process_top().adj_begin()


def adj_end():
    return process_top().adj_end()


def jump(goto: label):
    return process_top().jump(goto)


def branch(test: future, then: label, otherwise: label):
    return process_top().branch(test, then, otherwise)


def X(q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_PAULI_X, qubit, 0.0)


def Y(q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_PAULI_Y, qubit, 0.0)


def Z(q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_PAULI_Z, qubit, 0.0)


def H(q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_HADAMARD, qubit, 0.0)


def S(q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, pi/2)


def SD(q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, -pi/2)


def T(q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, pi/4)


def TD(q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, -pi/4)


def phase(lambda_, q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_PHASE, qubit, lambda_)


def RX(theta, q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_ROTATION_X, qubit, theta)


def RY(theta, q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_ROTATION_Y, qubit, theta)


def RZ(theta, q: quant):
    for qubit in q.qubits:
        process_top().gate(KET_ROTATION_Z, qubit, theta)
