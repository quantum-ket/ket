# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ket import Process, measure, dump, CNOT
from ket.qint import Qint, Qreal


def test_qint_initialization():
    """Test Qint initialization with a classical integer."""
    p = Process()
    q = p.alloc(4)

    # Initialize Qint with value 5
    qi = Qint(q, 5)

    res = measure(qi)

    assert res.value == 5


def test_qreal_initialization():
    """Test Qreal fixed-point initialization."""
    p = Process()
    q = p.alloc(4)

    # Exponent 2 means value is scaled by 2**2 (4).
    # Storing 1.5 classically means 1.5 * 4 = 6 internally.
    qr = Qreal(q, exp=2, number=1.5)

    # Dump to verify postprocessing works correctly
    state = dump(qr)

    # Only the state corresponding to 1.5 should have probability 1.0
    probs = state.probability
    # Note: the internal state is 6, but postprocessing converts it to float
    internal_state = list(probs.keys())[0]
    assert internal_state == 6
    assert qr.postprocessing()(internal_state) == 1.5


def test_qint_addition():
    """Test addition between Qint and integer."""
    p = Process()
    q1 = p.alloc(5)
    q2 = p.alloc(5)
    qres = p.alloc(5).as_int()

    qi1 = Qint(q1, 3)
    qi2 = Qint(q2, 4)

    # Quantum-Quantum addition
    qi3 = qi1 + qi2
    # Quantum-Classical addition
    qi4 = qi3 + 2

    qres += qi4

    res = measure(qres)

    assert res.value == 9


def test_qint_auxiliary_measurement_protection():
    """Test that measuring the direct result of an expression fails due to auxiliary protection.

    Operations like `a + b` allocate temporary auxiliary qubits. Measuring them directly
    violates the Safe Uncomputation Pattern.
    """
    p = Process()
    q1 = Qint(p.alloc(4), 3)
    q2 = Qint(p.alloc(4), 4)

    # qi_aux is backed by an auxiliary register managed by `undo`
    qi_aux = q1 + q2

    # Trying to measure it directly MUST raise a ValueError
    with pytest.raises(ValueError, match="Auxiliary qubits cannot be measured"):
        measure(qi_aux)


def test_qint_twos_complement_overflow():
    """Test that Qint arithmetic correctly overflows using two's complement representation.

    A 4-bit signed integer can represent values from -8 to 7.
    Adding 4 + 5 = 9. In 4-bit binary, 9 is 1001, which represents -7 in two's complement.
    """
    p = Process()

    # Allocate 4 qubits (max value 7, min value -8)
    q_overflow = p.alloc(4).as_int()
    q_overflow += 4
    q_overflow += 5

    res_overflow = measure(q_overflow)

    # The bitstring will be 1001, which Qint postprocessing interprets as -7
    assert res_overflow.value == -7


def test_qint_safe_addition():
    """Test that adding into a properly sized, non-auxiliary Qint works as expected."""
    p = Process()

    # 5 bits can represent -16 to 15.
    q_safe = p.alloc(5).as_int(0)

    q1 = Qint(p.alloc(5), 3)
    q2 = Qint(p.alloc(5), 4)

    # Addition creates an auxiliary qubit...
    qi_aux = q1 + q2

    # ...but accumulating it into a clean, non-auxiliary register makes it safe to measure
    q_safe += qi_aux
    q_safe += 2

    res = measure(q_safe)

    # 3 + 4 + 2 = 9 (fits in 5 bits)
    assert res.value == 9


def test_qint_subtraction():
    """Test subtraction operation and negative values."""
    p = Process()
    q = p.alloc(4)
    qres = p.alloc(4).as_int()

    qi = Qint(q, 5)
    result = qi - 2

    qres += result

    res = measure(qres)

    assert res.value == 3


def test_qint_size_exception():
    """Test that arithmetic operations catch mismatched sizes."""
    p = Process()
    lhs = Qint(p.alloc(3), 0)
    rhs = Qint(p.alloc(4), 5)  # RHS is larger than LHS

    with pytest.raises(
        RuntimeError,
        match="The number of qubits in the right-hand side must be less than or equal to the left-hand side",
    ):
        _ = lhs + rhs

    with pytest.raises(
        RuntimeError,
        match="The number of bits in the right-hand side must be less than or equal to the left-hand side",
    ):
        _ = lhs + 15  # 15 needs 4 bits, LHS only has 3


def test_qint_type_exception():
    """Test type checking on overloaded operators to avoid Python native crashes."""
    p = Process()
    qi = Qint(p.alloc(3), 2)

    # Trying to add a string should fall back to NotImplemented and raise TypeError natively
    with pytest.raises(TypeError):
        _ = qi + "invalid_type"


def test_qint_comparisons():
    """Test equality and relational operators."""
    p = Process()
    q1 = p.alloc(3)
    qres_eq_res = p.alloc()
    qres_gt_res = p.alloc()
    qres_lt_res = p.alloc()

    qi1 = Qint(q1, 2)

    # Comparisons return a Quant containing a single boolean qubit
    eq_res = qi1 == 2
    gt_res = qi1 > 1
    lt_res = qi1 < 1

    CNOT(eq_res, qres_eq_res)
    CNOT(gt_res, qres_gt_res)
    CNOT(lt_res, qres_lt_res)

    res_eq = measure(qres_eq_res)
    res_gt = measure(qres_gt_res)
    res_lt = measure(qres_lt_res)

    assert res_eq.value == 1  # 2 == 2 (True)
    assert res_gt.value == 1  # 2 > 1 (True)
    assert res_lt.value == 0  # 2 < 1 (False)


def _trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


@pytest.mark.parametrize("a", [-8, -7, -3, 0, 5, 7])
@pytest.mark.parametrize("b", [-4, -3, -1, 1, 2, 3])
def test_qint_division(a, b):
    """Quantum division truncates toward zero, for any signs."""
    p = Process(num_qubits=40)
    qa = Qint(p.alloc(4), a)
    qb = Qint(p.alloc(3), b)
    qres = p.alloc(4).as_int()

    qres += qa / qb

    expected = _trunc_div(a, b)
    if expected == 8:  # -8 / -1 wraps around in 4 bits
        expected = -8
    assert measure(qres).value == expected


@pytest.mark.parametrize("b", [1, 2, 3, -2, -5])
def test_qint_division_by_int(b):
    p = Process()
    qa = Qint(p.alloc(5), -13)
    qres = p.alloc(5).as_int()

    qres += qa / b

    assert measure(qres).value == _trunc_div(-13, b)


def test_qreal_division():
    """Fixed-point division keeps the exponent of the dividend."""
    p = Process(num_qubits=64)
    x = p.alloc(7).as_real(2, 3.0)
    y = p.alloc(4).as_real(1, 1.5)
    res = p.alloc(7).as_real(2)

    res += x / y
    res += x / 0.5

    assert measure(res).value == 2.0 + 6.0


def test_qint_division_by_zero():
    p = Process()
    with pytest.raises(ZeroDivisionError):
        _ = p.alloc(3).as_int(2) / 0


@pytest.mark.parametrize("a", [-8, -3, 0, 5])
@pytest.mark.parametrize("b", [-4, -1, 0, 3])
def test_qint_add_shorter_signed(a, b):
    """A shorter negative register is sign-extended, not zero-extended."""
    p = Process(num_qubits=40)
    qa = Qint(p.alloc(6), a)
    qa += Qint(p.alloc(3), b)
    assert measure(qa).value == a + b

    p = Process(num_qubits=40)
    qa = Qint(p.alloc(6), a)
    qa -= Qint(p.alloc(3), b)
    assert measure(qa).value == a - b

    p = Process(num_qubits=40)
    qres = p.alloc(6).as_int()
    qres += Qint(p.alloc(6), a) * Qint(p.alloc(3), b)
    assert measure(qres).value == (a * b + 32) % 64 - 32  # wraps in 6 bits


def test_qreal_add_shorter_signed():
    p = Process()
    x = p.alloc(8).as_real(2, 1.25)
    x += p.alloc(4).as_real(1, -2.5)
    assert measure(x).value == -1.25
