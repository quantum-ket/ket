"""Unit tests for fermionic operator functionality in ket."""
# pylint: disable=invalid-name

# SPDX-FileCopyrightText: 2026 Maria Eduarda W. M. Vianna <mewmvianna@gmail.com>
# SPDX-FileCopyrightText: 2026 Erico Souza Teixeira <erico.teixeira@venturus.org.br>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ket


def test_dict_creation():
    """Test fermionic operator creation from dictionaries."""

    a0 = ket.Fermion({(0, 0): "+"})
    a1 = ket.Fermion({(0, 1): "-"})

    word = ket.Fermion(
        {
            (0, 0): "+",
            (1, 1): "-",
            (2, 2): "+",
        }
    )

    assert a0 == ket.CreateFermion(0)
    assert a1 == ket.AnnihilateFermion(1)

    expected_word = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
        * ket.CreateFermion(2)
    )

    assert word == expected_word


def test_dict_creation_with_spin():
    """Test fermionic operator creation with explicit spin."""

    a0_alpha = ket.Fermion({(0, 0, "a"): "+"})
    a0_beta = ket.Fermion({(0, 0, "b"): "-"})

    assert a0_alpha.get_spin()[(0, 0)] == "a"
    assert a0_beta.get_spin()[(0, 0)] == "b"


def test_invalid_operator():
    """Test rejection of invalid operator symbols."""

    with pytest.raises(ValueError):
        ket.Fermion({(0, 0): "ket"})


def test_basic():
    """Test creation and annihilation operators."""

    assert ket.CreateFermion(0) == ket.Fermion({(0, 0, "a"): "+"})
    assert ket.AnnihilateFermion(1) == ket.Fermion({(0, 1, "b"): "-"})


def test_invalid_orbital():
    """Test orbital validation."""

    with pytest.raises(TypeError):
        ket.CreateFermion("z")

    with pytest.raises(ValueError):
        ket.CreateFermion(-1)


def test_representation():
    """Test string representations."""

    word = ket.CreateFermion(0) * ket.AnnihilateFermion(1)

    assert str(word) == "a⁺(0) a(1)"
    assert repr(word).startswith("Fermion(")


def test_fermion_sentence_creation():
    """Test FermionSentence creation from dictionaries."""

    word = ket.CreateFermion(0) * ket.AnnihilateFermion(1)

    sentence = ket.FermionSentence(
        {
            word: 1.3,
            ket.AnnihilateFermion(1): 2.4,
        }
    )

    assert sentence[word] == 1.3
    assert sentence[ket.AnnihilateFermion(1)] == 2.4


def test_product():
    """Test fermionic operator multiplication."""

    product = ket.CreateFermion(0) * ket.AnnihilateFermion(1)

    expected = ket.Fermion(
        {
            (0, 0, "a"): "+",
            (1, 1, "b"): "-",
        }
    )

    assert product == expected


def test_linear_combination():
    """Test linear combinations of fermionic operators."""

    word = ket.CreateFermion(0) * ket.AnnihilateFermion(1)

    sentence = 1.3 * word + 2.4 * ket.AnnihilateFermion(1)

    assert sentence[word] == 1.3
    assert sentence[ket.AnnihilateFermion(1)] == 2.4


@pytest.mark.parametrize(
    ("orbital", "expected_spin"),
    [
        (0, "a"),
        (1, "b"),
        (2, "a"),
        (3, "b"),
    ],
)
def test_default_spin_rule(orbital, expected_spin):
    """Test default spin assignment."""

    op = ket.CreateFermion(orbital)

    spin_map = op.get_spin()
    key = next(iter(spin_map))

    assert spin_map[key] == expected_spin


def test_explicit_spin_override():
    """Test explicit spin override."""

    b0 = ket.CreateFermion(0, spin="b")
    b1 = ket.CreateFermion(1, spin="a")

    assert b0.get_spin()[(0, 0)] == "b"
    assert b1.get_spin()[(0, 1)] == "a"


def test_manual_spin_inference():
    """Test spin inference in manual constructor."""

    c0 = ket.Fermion({(0, 0): "+"})
    c1 = ket.Fermion({(0, 1): "-"})
    c2 = ket.Fermion({(0, 0, "b"): "+"})

    assert c0.get_spin()[(0, 0)] == "a"
    assert c1.get_spin()[(0, 1)] == "b"
    assert c2.get_spin()[(0, 0)] == "b"


def test_product_preserves_spin():
    """Test spin preservation under multiplication."""

    c0 = ket.Fermion({(0, 0): "+"})
    c1 = ket.Fermion({(0, 1): "-"})

    product = c0 * c1

    spin_map = product.get_spin()

    assert spin_map[(0, 0)] == "a"
    assert spin_map[(1, 1)] == "b"


def test_simplify_removes_small_terms():
    """Test removal of terms below tolerance."""

    a0 = ket.CreateFermion(0)
    a1 = ket.AnnihilateFermion(1)

    sentence = ket.FermionSentence(
        {
            a0: 1.0,
            a1: 1e-12,
        }
    )

    sentence.simplify()

    assert a0 in sentence
    assert a1 not in sentence


def test_simplify_boundary_case():
    """Test simplify boundary tolerance."""

    a2 = ket.CreateFermion(2)

    sentence = ket.FermionSentence(
        {
            a2: 1e-8,
        }
    )

    sentence.simplify()

    assert a2 not in sentence

def test_single_orbital_number_operator():
    """Test number operator for a single orbital."""

    operator = ket.number_operator(4, orbital=2)

    expected_term = (
        ket.CreateFermion(2)
        * ket.AnnihilateFermion(2)
    )

    assert expected_term in operator
    assert len(operator) == 1


def test_total_number_operator():
    """Test total number operator construction."""

    operator = ket.number_operator(4)

    assert len(operator) == 4

    for orbital in range(4):
        expected_term = (
            ket.CreateFermion(orbital)
            * ket.AnnihilateFermion(orbital)
        )

        assert expected_term in operator


def test_number_operator_invalid_n_orbitals():
    """Test invalid number of orbitals."""

    with pytest.raises(ValueError):
        ket.number_operator(-1)


def test_number_operator_invalid_orbital():
    """Test invalid orbital index."""

    with pytest.raises(ValueError):
        ket.number_operator(4, orbital=10)


def test_particle_number_conservation_one_body():
    """Test particle-number conservation for one-body operators."""

    term = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert sentence.conserves_particle_number()


def test_particle_number_conservation_two_body():
    """Test particle-number conservation for two-body operators."""

    term = (
        ket.CreateFermion(0)
        * ket.CreateFermion(1)
        * ket.AnnihilateFermion(2)
        * ket.AnnihilateFermion(3)
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert sentence.conserves_particle_number()


def test_particle_number_non_conserving():
    """Test detection of non-conserving operators."""

    term = ket.CreateFermion(0)

    sentence = ket.FermionSentence({term: 1.0})

    assert not sentence.conserves_particle_number()


def test_particle_number_mixed_sentence():
    """Test mixed sentence containing non-conserving terms."""

    conserving = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    non_conserving = ket.CreateFermion(0)

    sentence = ket.FermionSentence(
        {
            conserving: 1.0,
            non_conserving: 2.0,
        }
    )

    assert not sentence.conserves_particle_number()


def test_spin_z_conservation_alpha():
    """Test spin-z conservation for alpha-spin operators."""

    term = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(2)
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert sentence.conserves_spin_z()


def test_spin_z_conservation_beta():
    """Test spin-z conservation for beta-spin operators."""

    term = (
        ket.CreateFermion(1)
        * ket.AnnihilateFermion(3)
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert sentence.conserves_spin_z()


def test_spin_z_non_conserving():
    """Test spin-flip operator detection."""

    term = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert not sentence.conserves_spin_z()


def test_spin_z_user_defined_override():
    """Test spin-z conservation with explicit spin labels."""

    term = (
        ket.CreateFermion(0, spin="b")
        * ket.AnnihilateFermion(2, spin="b")
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert sentence.conserves_spin_z()


def test_spin_z_mixed_sentence():
    """Test mixed sentence containing spin-flip terms."""

    conserving = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(2)
    )

    spin_flip = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    sentence = ket.FermionSentence(
        {
            conserving: 1.0,
            spin_flip: 2.0,
        }
    )

    assert not sentence.conserves_spin_z()

def test_two_body_number_conserving_one_body():
    """Test one-body number-conserving operators."""

    term = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert sentence.is_two_body_number_conserving()


def test_two_body_number_conserving_two_body():
    """Test two-body number-conserving operators."""

    term = (
        ket.CreateFermion(0)
        * ket.CreateFermion(1)
        * ket.AnnihilateFermion(2)
        * ket.AnnihilateFermion(3)
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert sentence.is_two_body_number_conserving()


def test_two_body_number_conserving_rejects_three_body():
    """Test rejection of operators beyond two-body."""

    term = (
        ket.CreateFermion(0)
        * ket.CreateFermion(1)
        * ket.CreateFermion(2)
        * ket.AnnihilateFermion(3)
        * ket.AnnihilateFermion(4)
        * ket.AnnihilateFermion(5)
    )

    sentence = ket.FermionSentence({term: 1.0})

    assert not sentence.is_two_body_number_conserving()


def test_two_body_number_conserving_rejects_non_conserving():
    """Test rejection of particle-number non-conserving operators."""

    sentence = ket.FermionSentence(
        {
            ket.CreateFermion(0): 1.0,
        }
    )

    assert not sentence.is_two_body_number_conserving()


def test_fermion_adjoint_creation():
    """Test adjoint of a creation operator."""

    op = ket.CreateFermion(0)

    assert op.adjoint() == ket.AnnihilateFermion(0)


def test_fermion_adjoint_annihilation():
    """Test adjoint of an annihilation operator."""

    op = ket.AnnihilateFermion(1)

    assert op.adjoint() == ket.CreateFermion(1)


def test_fermion_adjoint_reverses_order():
    """Test that adjoint reverses operator order."""

    op = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    expected = (
        ket.CreateFermion(1)
        * ket.AnnihilateFermion(0)
    )

    assert op.adjoint() == expected


def test_double_adjoint_fermion():
    """Test that applying adjoint twice returns the original operator."""

    op = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
        * ket.CreateFermion(2)
    )

    assert op.adjoint().adjoint() == op


def test_sentence_adjoint_real_coefficients():
    """Test adjoint of a sentence with real coefficients."""

    term = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    sentence = ket.FermionSentence({term: 2.0})

    adjoint = sentence.adjoint()

    assert adjoint[term.adjoint()] == 2.0


def test_sentence_adjoint_complex_coefficients():
    """Test complex conjugation under adjoint."""

    term = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    sentence = ket.FermionSentence({term: 1 + 2j})

    adjoint = sentence.adjoint()

    assert adjoint[term.adjoint()] == 1 - 2j


def test_double_adjoint_sentence():
    """Test that applying adjoint twice returns the original sentence."""

    term = (
        ket.CreateFermion(0)
        * ket.AnnihilateFermion(1)
    )

    sentence = ket.FermionSentence(
        {
            term: 3 - 4j,
        }
    )

    assert sentence.adjoint().adjoint() == sentence
