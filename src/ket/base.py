"""Basic Ket type definitions."""
from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import c_int32, c_size_t, c_uint8
from json import loads
from random import Random
from typing import Literal, Optional

from .clib.libket import Process as LibketProcess, API
from .clib.kbw import get_simulator

__all__ = [
    "Process",
    "Quant",
    "measure",
    "sample",
    "dump",
    "Pauli",
    "exp_value",
]


class Process(LibketProcess):
    """Quantum process for handling the quantum circuit creation and execution."""

    def __init__(
        self,
        configuration=None,
        num_qubits: Optional[int] = None,
        simulator: Optional[Literal["sparse", "dense"]] = None,
        execution: Optional[Literal["live", "batch"]] = None,
    ):
        if configuration is not None and any(
            map(lambda a: a is not None, [num_qubits, simulator, execution])
        ):
            raise ValueError(
                "Cannot specify num_qubits, simulator or execution if configuration is provided"
            )

        if configuration is not None:
            super().__init__(configuration)
        else:
            simulator = "sparse" if simulator is None else simulator
            num_qubits = (
                (32 if simulator == "sparse" else 12)
                if num_qubits is None
                else num_qubits
            )
            super().__init__(
                get_simulator(
                    num_qubits=num_qubits,
                    simulator=simulator,
                    execution="live" if execution is None else execution,
                )
            )

        self._metadata_buffer_size = 512
        self._metadata_buffer = (c_uint8 * self._metadata_buffer_size)()
        self._instructions_buffer_size = 2048
        self._instructions_buffer = (c_uint8 * self._instructions_buffer_size)()

    def alloc(self, num_qubit: int = 1) -> Quant:
        """Allocate ``num_qubit`` qubits and return a :class:`~ket.quant` object."""

        if num_qubit < 1:
            raise ValueError("Cannot allocate less than 1 qubit")

        qubits_index = [self.allocate_qubit().value for _ in range(num_qubit)]
        return Quant(qubits=qubits_index, process=self)

    def execute(self):
        """Force the quantum circuit execution."""
        self.prepare_for_execution()

    def _get_ket_process(self):
        return self

    def get_instructions(self) -> dict:
        """Get the quantum instructions from the process."""
        write_size = self.instructions_json(
            self._instructions_buffer, self._instructions_buffer_size
        )
        if write_size.value > self._instructions_buffer_size:
            self._instructions_buffer_size = write_size.value + 1
            self._instructions_buffer = (c_uint8 * self._instructions_buffer_size)()
            return self.get_instructions()

        return loads(bytearray(self._instructions_buffer[: write_size.value]))

    def get_metadata(self) -> dict:
        """Get the metadata from the process."""
        write_size = self.metadata_json(
            self._metadata_buffer, self._metadata_buffer_size
        )
        if write_size.value > self._metadata_buffer_size:
            self._metadata_buffer_size = write_size.value + 1
            self._metadata_buffer = (c_uint8 * self._metadata_buffer_size)()
            return self.get_metadata()

        return loads(bytearray(self._metadata_buffer[: write_size.value]))

    def __repr__(self) -> str:
        return f"<Ket 'Process' id={hex(id(self))}>"


class Quant:
    """List of qubits."""

    def __init__(self, *, qubits, process):
        self.qubits = qubits
        self.process = process

    def _get_ket_process(self):
        return self.process

    def __add__(self, other: Quant) -> Quant:
        if self.process is not other.process:
            raise ValueError("Cannot concatenate qubits from different processes")
        return Quant(qubits=self.qubits + other.qubits, process=self.process)

    def at(self, index: list[int]) -> Quant:
        r"""Return qubits at ``index``

        Create a new :class:`~ket.base.quant` with the qubit references at the
        position defined by the ``index`` list.

        :Example:

        .. code-block:: ket

            q = quant(20)
            odd = q.at(range(1, len(q), 2)) # = q[1::2]

        Args:
            index: List of indexes.
        """

        return Quant(qubits=[self.qubits[i] for i in index], process=self.process)

    def free(self):
        r"""Free the qubits

        All qubits must be at the state :math:`\left|0\right>` before the call.

        Warning:
            No check is applied to see if the qubits are at state
            :math:`\left|0\right>`.
        """

        for qubit in self.qubits:
            self.process.free_qubit(qubit)

    def is_free(self) -> bool:
        """Return ``True`` when all qubits are free"""
        return all(
            not self.process.get_qubit_status(qubit)[0].value for qubit in self.qubits
        )

    def __reversed__(self):
        return Quant(qubits=list(reversed(self.qubits)), process=self.process)

    def __getitem__(self, key):
        qubits = self.qubits.__getitem__(key)
        return Quant(
            qubits=qubits if isinstance(qubits, list) else [qubits],
            process=self.process,
        )

    class Iter:
        """Qubits iterator"""

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
        return self.Iter(self)

    def __enter__(self):
        return self

    def __exit__(
        self, type, value, tb
    ):  # pylint: disable=redefined-builtin, invalid-name
        if not self.is_free():
            raise RuntimeError("non-free quant at the end of scope")

    def __len__(self):
        return len(self.qubits)

    def __repr__(self):
        return f"<Ket 'Quant' {self.qubits} pid={hex(id(self.process))}>"


class Measurement:
    """Quantum measurement result."""

    def __init__(self, qubits: Quant):
        self.process = qubits.process
        self.qubits = [qubits[i : i + 64] for i in range(0, len(qubits), 64)]
        self.indexes = [
            self.process.measure(
                (c_size_t * len(qubit.qubits))(*qubit.qubits), len(qubit.qubits)
            ).value
            for qubit in self.qubits
        ]
        self._value = None

    def _get_ket_process(self):
        return self.process

    def _check(self):
        if self._value is None:
            available, values = zip(
                *(self.process.get_measurement(index) for index in self.indexes)
            )
            if all(map(lambda a: a.value, available)):
                self._value = 0
                for value, qubit in zip(values, self.qubits):
                    self._value <<= len(qubit)
                    self._value |= value.value

    @property
    def value(self) -> int | None:
        """Return the measurement value if available."""
        self._check()
        return self._value

    def __repr__(self):
        return (
            f"<Ket 'Measurement' indexes={self.indexes}, "
            f"value={self.value}, pid={hex(id(self.process))}>"
        )


def measure(qubits: Quant) -> Measurement:
    """Measure the qubits and return a :class:`~ket.base.measurement` object."""
    return Measurement(qubits)


class QuantumState:
    """Create a snapshot with the current quantum state of ``qubits``

    Gathering any information from a :class:`~ket.base.dump` triggers the quantum execution.

    :Example:

    .. code-block:: ket

        a, b = quant(2)
        with around(cnot(H, I), a, b):
            Y(a)
            inside = dump(a+b)
        outside = dump(a+b)

        print('inside:')
        print(inside.show())
        # inside:
        # |01⟩    (50.00%)
        #          -0.707107i     ≅     -i/√2
        # |10⟩    (50.00%)
        #           0.707107i     ≅      i/√2
        print('outside:')
        print(outside.show())
        # outside:
        # |11⟩    (100.00%)
        #          -1.000000i     ≅     -i/√1

    :param qubits: Qubits to dump.
    """

    def __init__(self, qubits: Quant):
        self.qubits = qubits
        self.process = qubits.process
        self.index = self.process.dump(
            (c_size_t * len(qubits.qubits))(*qubits.qubits), len(qubits.qubits)
        ).value
        self.size = len(qubits)
        self._states = None

    def _get_ket_process(self):
        return self.process

    def _check(self):
        if self._states is None:
            available, size = self.process.get_dump_size(self.index)
            if available.value:
                self._states = {}
                for i in range(size.value):
                    state, state_size, amp_real, amp_imag = self.process.get_dump(
                        self.index, i
                    )
                    state = int(
                        "".join(f"{state[j]:064b}" for j in range(state_size.value)), 2
                    )
                    amplitude = complex(amp_real.value, amp_imag.value)
                    self._states[state] = amplitude

    @property
    def states(self) -> dict[int, complex] | None:
        """Get the quantum state

        This function returns a ``dict`` that maps base states to probability amplitude.
        """

        self._check()
        return self._states

    @property
    def probabilities(self) -> dict[int, float] | None:
        """List of measurement probabilities"""

        self._check()
        if self._states is None:
            return None
        return dict(map(lambda k, v: (k, abs(v * v)), self._states.items()))

    def sample(self, shots=4096, seed=None) -> dict[int, int] | None:
        """Get the quantum execution shots

        If the dump variable does not hold the result of shots execution,
        the parameters `shots` and `seed` are used to generate the sample.

        Args:
            shots: Number of shots (used if the dump does not store the result of shots execution)
            seed: Seed for the RNG (used if the dump does not store the result of shots execution)
        """

        self._check()
        if self._states is None:
            return None

        rng = Random(seed)
        shots = rng.choices(list(self.states), list(self.probabilities), k=shots)
        result = {}
        for state in shots:
            if state not in result:
                result[state] = 1
            else:
                result[state] += 1
        return result

    def show(self, format_str: str | None = None) -> str | None:
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
            # |87381⟩ (50.00%)
            #  0.707107               ≅      1/√2
            # |436906⟩        (50.00%)
            #  0.707107               ≅      1/√2
            print(d.show('b'))
            # |0010101010101010101⟩   (50.00%)
            #  0.707107               ≅      1/√2
            # |1101010101010101010⟩   (50.00%)
            #  0.707107               ≅      1/√2
            print(d.show('i4'))
            # |2⟩|101010101010101⟩    (50.00%)
            #  0.707107               ≅      1/√2
            # |13⟩|010101010101010⟩   (50.00%)
            #  0.707107               ≅      1/√2
            print(d.show('b5:i4'))
            # |00101⟩|5⟩|0101010101⟩  (50.00%)
            #  0.707107               ≅      1/√2
            # |11010⟩|10⟩|1010101010⟩ (50.00%)
            #  0.707107               ≅      1/√2

        Args:
            format: Format string that matches ``(i|b)\d*(:(i|b)\d+)*``.
        """

        self._check()
        if self._states is None:
            return None

        if format_str is not None:
            if format_str in ("b", "i"):
                format_str += str(self.size)
            fmt = []
            count = 0
            for b, size in map(lambda f: (f[0], int(f[1:])), format_str.split(":")):
                fmt.append((b, count, count + size))
                count += size
            if count < self.size:
                fmt.append(("b", count, self.size))
        else:
            fmt = [("b", 0, self.size)]

        def fmt_ket(state, begin, end, f):
            return (
                f"|{state[begin:end]}⟩"
                if f == "b"
                else f"|{int(state[begin:end], base=2)}⟩"
            )

        def state_amp_str(state, amp):
            dump_str = "".join(
                fmt_ket(f"{state:0{self.size}b}", b, e, f) for f, b, e in fmt
            )
            dump_str += f"\t({100*abs(amp)**2:.2f}%)\n"
            real = abs(amp.real) > 1e-10
            real_l0 = amp.real < 0

            imag = abs(amp.imag) > 1e-10
            imag_l0 = amp.imag < 0

            sqrt_dem = 1 / abs(amp) ** 2
            use_sqrt = abs(round(sqrt_dem) - sqrt_dem) < 0.001
            use_sqrt = use_sqrt and (
                (abs(abs(amp.real) - abs(amp.imag)) < 1e-6) or (real != imag)
            )
            sqrt_dem = f"/√{round(1/abs(amp)**2)}"

            if real and imag:
                sqrt_dem = f"/√{round(2*(1/abs(amp)**2))}"
                sqrt_num = ("(-1" if real_l0 else " (1") + ("-i" if imag_l0 else "+i")
                sqrt_str = (
                    f"\t≅ {sqrt_num}){sqrt_dem}"
                    if use_sqrt and (abs(amp.real) - abs(amp.real) < 1e-10)
                    else ""
                )
                dump_str += f"{amp.real:9.6f}{amp.imag:+.6f}i" + sqrt_str
            elif real:
                sqrt_num = "  -1" if real_l0 else "   1"
                sqrt_str = f"\t≅   {sqrt_num}{sqrt_dem}" if use_sqrt else ""
                dump_str += f"{amp.real:9.6f}       " + sqrt_str
            else:
                sqrt_num = "  -i" if imag_l0 else "   i"
                sqrt_str = f"\t≅   {sqrt_num}{sqrt_dem}" if use_sqrt else ""
                dump_str += f" {amp.imag:17.6f}i" + sqrt_str

            return dump_str

        return "\n".join(
            state_amp_str(state, amp)
            for state, amp in sorted(self._states.items(), key=lambda k: k[0])
        )

    def __repr__(self):
        return f"<Ket 'QuantumState' index={self.index}, pid={hex(id(self.process))}>"


def dump(qubits: Quant) -> QuantumState:
    """Get the quantum state"""
    return QuantumState(qubits)


class Samples:
    """Quantum state measurement samples."""

    def __init__(self, qubits: Quant, shots: int = 2048):
        self.qubits = qubits
        self.process = qubits.process
        self.index = self.process.sample(
            (c_size_t * len(qubits.qubits))(*qubits.qubits), len(qubits.qubits), shots
        ).value
        self._value = None

    def _check(self):
        if self._value is None:
            (
                available,
                states,
                count,
                size,
            ) = self.process.get_sample(self.index)
            if available.value:
                self._value = dict(zip(states[: size.value], count[: size.value]))

    @property
    def value(self) -> dict[int, int] | None:
        """Get the measurement samples"""
        self._check()
        return self._value

    def __repr__(self) -> str:
        return f"<Ket 'Samples' index={self.index}, pid={hex(id(self.process))}>"


def sample(qubits: Quant, shots: int = 2048) -> Samples:
    """Get the quantum state measurement samples"""
    return Samples(qubits, shots)


class Pauli:
    """Pauli operator for Hamiltonian creation."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pauli: Literal["X", "Y", "Z"],
        qubits: Quant,
        *,
        process=None,
        pauli_list: list[str] | None = None,
        qubits_list: list[Quant] | None = None,
        coef: float | None = None,
    ):
        self.process = process if process is not None else qubits.process
        self.pauli_list = pauli_list if pauli_list is not None else [pauli]
        self.qubits_list = qubits_list if qubits_list is not None else [qubits]
        self.coef = 1.0 if coef is None else coef

    def _flat(self) -> tuple[list[str], list[Quant]]:
        pauli_list = []
        qubits_list = []
        for pauli, qubits in zip(self.pauli_list, self.qubits_list):
            pauli_list += [pauli] * len(qubits.qubits)
            qubits_list += qubits.qubits
        return pauli_list, qubits_list

    def __mul__(self, other: float | Pauli) -> Pauli:
        if isinstance(other, float):
            return Pauli(
                None,
                None,
                process=self.process,
                pauli_list=self.pauli_list,
                qubits_list=self.qubits_list,
                coef=self.coef * other,
            )

        if self.process is not other.process:
            raise ValueError("different Ket processes")

        return Pauli(
            None,
            None,
            process=self.process,
            pauli_list=self.pauli_list + other.pauli_list,
            qubits_list=self.qubits_list + other.qubits_list,
            coef=self.coef * other.coef,
        )

    def __rmul__(self, other: float) -> Pauli:
        return Pauli(
            None,
            None,
            process=self.process,
            pauli_list=self.pauli_list,
            qubits_list=self.qubits_list,
            coef=self.coef * other,
        )

    def __add__(self, other) -> Hamiltonian:
        if self.process is not other.process:
            raise ValueError("different Ket processes")

        return Hamiltonian([self, other], process=self.process)

    def __repr__(self) -> str:
        return f"{self.coef}*" + "".join(
            "".join(f"{pauli}{qubit}" for qubit in qubits.qubits)
            for pauli, qubits in zip(self.pauli_list, self.qubits_list)
        )


class Hamiltonian:
    """Hamiltonian for expected value calculation."""

    def __init__(self, pauli_products: list[Pauli], process):
        self.process = process
        self.pauli_products = pauli_products

    def __add__(self, other: Hamiltonian | Pauli) -> Hamiltonian:
        if isinstance(other, Pauli):
            other = Hamiltonian([other], self.process)
            return self + other

        if self.process is not other.process:
            raise ValueError("different Ket processes")

        return Hamiltonian(self.pauli_products + other.pauli_products, self.process)

    def __mul__(self, other: float) -> Hamiltonian:
        return Hamiltonian(
            [p * other for p in self.pauli_products], process=self.process
        )

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return (
            f"<Ket 'Hamiltonian' {' + '.join(str(p) for p in self.pauli_products)}, "
            f"pid={hex(id(self.process))}>"
        )


class ExpValue:
    """Expected value for a quantum state."""

    pauli_map = {"X": 1, "Y": 2, "Z": 3}

    def __init__(self, hamiltonian: Hamiltonian | Pauli):
        if isinstance(hamiltonian, Pauli):
            hamiltonian = Hamiltonian([hamiltonian], process=hamiltonian.process)

        self.process = hamiltonian.process

        hamiltonian_ptr = API["ket_hamiltonian_new"]()
        for pauli_product in hamiltonian.pauli_products:
            pauli, qubits = pauli_product._flat()
            pauli = [self.pauli_map[p] for p in pauli]
            API["ket_hamiltonian_add"](
                hamiltonian_ptr,
                (c_int32 * len(pauli))(*pauli),
                len(pauli),
                (c_size_t * len(qubits))(*qubits),
                len(qubits),
                pauli_product.coef,
            )

        self.index = self.process.exp_value(hamiltonian_ptr).value
        self._value = None

    def _check(self):
        if self._value is None:
            available, value = self.process.get_exp_value(self.index)
            if available.value:
                self._value = value.value

    @property
    def value(self) -> float | None:
        """Expected value for a quantum state."""
        self._check()
        return self._value

    def __repr__(self) -> str:
        return f"<Ket 'ExpValue' value={self.value}, pid={hex(id(self.process))}>"


def exp_value(hamiltonian: Hamiltonian | Pauli) -> ExpValue:
    """Expected value for a quantum state."""
    return ExpValue(hamiltonian)
