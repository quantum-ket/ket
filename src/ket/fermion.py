"""Fermionic operators and utilities.

This module provides fermionic creation and annihilation operators,
fermionic operator sentences, and utilities for particle number
and spin conservation checks.
"""

# SPDX-FileCopyrightText: 2026 Maria Eduarda W. M. Vianna <mewmvianna@gmail.com>
# SPDX-FileCopyrightText: 2026 Erico Souza Teixeira <erico.teixeira@venturus.org.br>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, protected-access

from copy import copy

__all__ = [
    "Fermion",
    "FermionSentence",
    "CreateFermion",
    "AnnihilateFermion",
    "number_operator",
]


# ======================================================
# SPIN RULE
# ======================================================


def _resolve_spin(orbital: int, spin: str | None):
    if spin is not None:
        if spin not in ("a", "b"):
            raise ValueError("spin must be 'a' or 'b'")
        return spin

    return "a" if orbital % 2 == 0 else "b"


# ======================================================
# WORD (PRODUCT)
# ======================================================


class Fermion(dict):
    """Fermionic operator product.

    Represents an ordered product of fermionic creation and
    annihilation operators.
    """

    def __init__(self, operators: dict):

        if not isinstance(operators, dict):
            raise TypeError("Fermion must be initialized with dict")

        for key, op in operators.items():

            if len(key) == 2:
                pos, orb = key
                spin = None

            elif len(key) == 3:
                pos, orb, spin = key

            else:
                raise ValueError("key must be (pos, orb) or (pos, orb, spin)")

            if not isinstance(orb, int):
                raise TypeError("Orbital must be int")

            if orb < 0:
                raise ValueError("Orbital must be non-negative")

            if spin is None:
                spin = _resolve_spin(orb, None)

            if spin not in ("a", "b"):
                raise ValueError("Invalid spin")

            if op not in ("+", "-"):
                raise ValueError("Operator must be '+' or '-'")

            self[(pos, orb, spin)] = op

    def __hash__(self):
        return hash(frozenset(self.items()))

    def __str__(self):
        if not self:
            return "I"

        out = []
        for (_, orb, _), op in sorted(self.items()):
            out.append(f"a⁺({orb})" if op == "+" else f"a({orb})")

        return " ".join(out)

    def __repr__(self):
        return f"Fermion({dict(self)})"

    def _repr_latex_(self) -> str:
        """Default Jupyter LaTeX representation (no spin)."""
        return f"${self._latex_core(spin=False)}$"

    def _latex_core(self, spin=False) -> str:
        """Internal LaTeX builder (no math delimiters)."""

        if not self:
            return "I"

        out = []

        for (_, orb, sp), op in sorted(self.items()):

            idx = f"{orb},{sp}" if spin else f"{orb}"

            if op == "+":
                out.append(rf"a^{{\dagger}}_{{{idx}}}")
            else:
                out.append(rf"a_{{{idx}}}")

        return " ".join(out)

    def show(self, spin=False):
        """Return a string representation of the operator.

        Args:
            spin: Whether to include spin labels.

        Returns:
            Human-readable fermionic operator string.
        """
        if not self:
            return "I"

        out = []
        for (_, orb, sp), op in sorted(self.items()):
            if spin:
                out.append(f"a⁺({orb},{sp})" if op == "+" else f"a({orb},{sp})")
            else:
                out.append(f"a⁺({orb})" if op == "+" else f"a({orb})")

        return " ".join(out)

    def get_spin(self):
        """Return the spin associated with each operator."""
        return {(pos, orb): spin for (pos, orb, spin) in self.keys()}

    def adjoint(self):
        """Return the Hermitian adjoint of the fermionic operator."""
        new_operators = {}
        ordered_operators = sorted(self.items())
        reversed_operators = reversed(ordered_operators)

        for new_pos, ((_, orb, spin), op) in enumerate(reversed_operators):

            if op == "+":
                new_op = "-"
            elif op == "-":
                new_op = "+"
            else:
                raise ValueError(f"Invalid operator: {op}")

            new_operators[(new_pos, orb, spin)] = new_op

        return Fermion(new_operators)

    def __mul__(self, other):

        if isinstance(other, Fermion):

            new = dict(self)

            offset = len(self)

            for (pos, orb, spin), op in other.items():
                new[(pos + offset, orb, spin)] = op

            return Fermion(new)

        if isinstance(other, FermionSentence):
            return FermionSentence({self: 1}) * other

        raise TypeError("invalid multiplication")

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return FermionSentence({self: other})
        return self * other

    def __add__(self, other):
        """Promote Fermion + Fermion/FermionSentence → FermionSentence."""

        if isinstance(other, Fermion):
            return FermionSentence({self: 1, other: 1})

        if isinstance(other, FermionSentence):
            return FermionSentence({self: 1}) + other

        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        if isinstance(other, FermionSentence):
            return other + FermionSentence({self: 1})
        return self.__add__(other)

    def __sub__(self, other):
        return FermionSentence({self: 1}) - other


# ======================================================
# HIGH LEVEL API
# ======================================================


def CreateFermion(orbital: int, spin: str | None = None):
    """Create a fermionic creation operator.

    Args:
        orbital: Orbital index.
        spin: Optional spin label ("a" or "b").

    Returns:
        Fermionic creation operator.
    """
    spin = _resolve_spin(orbital, spin)
    return Fermion({(0, orbital, spin): "+"})


def AnnihilateFermion(orbital: int, spin: str | None = None):
    """Create a fermionic annihilation operator.

    Args:
        orbital: Orbital index.
        spin: Optional spin label ("a" or "b").

    Returns:
        Fermionic annihilation operator.
    """
    spin = _resolve_spin(orbital, spin)
    return Fermion({(0, orbital, spin): "-"})


def number_operator(n_orbitals: int, orbital: int | None = None):
    """Construct a fermionic number operator.

    Args:
        n_orbitals: Total number of orbitals.
        orbital: Orbital index. If None, returns the total
            number operator.

    Returns:
        Fermionic number operator as a FermionSentence.
    """
    if not isinstance(n_orbitals, int):
        raise TypeError("n_orbitals must be int")

    if n_orbitals < 0:
        raise ValueError("n_orbitals must be non-negative")

    if orbital is not None:

        if not isinstance(orbital, int):
            raise TypeError("orbital must be int")

        if orbital < 0:
            raise ValueError("orbital must be non-negative")

        if orbital >= n_orbitals:
            raise ValueError("orbital must be smaller than n_orbitals")

        term = CreateFermion(orbital) * AnnihilateFermion(orbital)

        return FermionSentence({term: 1})

    terms = {}

    for i in range(n_orbitals):
        term = CreateFermion(i) * AnnihilateFermion(i)
        terms[term] = 1

    return FermionSentence(terms)


# ======================================================
# LINEAR COMBINATION
# ======================================================


class FermionSentence(dict):
    """Linear combination of fermionic operator products."""

    def __str__(self):
        if not self:
            return "0 * I"
        return "\n+ ".join(f"{c} * {f}" for f, c in self.items())

    def __repr__(self):
        return f"FermionSentence({dict(self)})"

    def _latex_core(self) -> str:

        if not self:
            return "0"

        terms = []

        for op, coef in self.items():

            # coeficiente formatado
            if coef == 1:
                c = ""
            elif coef == -1:
                c = "-"
            else:
                c = f"{coef} "

            terms.append(f"{c}{op._latex_core()}")

        return " + ".join(terms).replace("+ -", "- ")

    def _repr_latex_(self) -> str:
        return f"${self._latex_core()}$"

    def __add__(self, other):

        if isinstance(other, Fermion):
            other = FermionSentence({other: 1})

        if not isinstance(other, FermionSentence):
            return NotImplemented

        result = copy(self)

        for k, v in other.items():
            result[k] = result.get(k, 0) + v

        return FermionSentence(result)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1) * other

    def adjoint(self):
        """Return the Hermitian adjoint of the fermionic sentence."""

        new_terms = {}

        for term, coef in self.items():
            new_terms[term.adjoint()] = complex(coef).conjugate()

        return FermionSentence(new_terms)

    def __mul__(self, other):

        # FermionSentence * FermionSentence
        if isinstance(other, FermionSentence):

            result = FermionSentence({})

            for t1, c1 in self.items():
                for t2, c2 in other.items():
                    new_term = t1 * t2
                    result[new_term] = result.get(new_term, 0) + c1 * c2

            return result

        # FermionSentence * Fermion
        if isinstance(other, Fermion):
            return self * FermionSentence({other: 1})

        # scalar
        return FermionSentence({k: v * other for k, v in self.items()})

    __rmul__ = __mul__

    def simplify(self, tol=1e-8) -> None:
        """Remove terms with small coefficients in place."""
        items = list(self.items())
        for term, coef in items:
            if abs(coef) <= tol:
                del self[term]

    def conserves_particle_number(self) -> bool:
        """Check whether the operator conserves particle number."""
        for term in self:

            n_creation = 0
            n_annihilation = 0

            for op in term.values():

                if op == "+":
                    n_creation += 1

                elif op == "-":
                    n_annihilation += 1

            if n_creation != n_annihilation:
                return False

        return True

    def conserves_spin_z(self) -> bool:
        """Check whether the operator conserves the z component of spin."""
        for term in self:

            alpha_creation = 0
            alpha_annihilation = 0

            beta_creation = 0
            beta_annihilation = 0

            for (_, _, spin), op in term.items():

                if spin == "a":

                    if op == "+":
                        alpha_creation += 1

                    elif op == "-":
                        alpha_annihilation += 1

                elif spin == "b":

                    if op == "+":
                        beta_creation += 1

                    elif op == "-":
                        beta_annihilation += 1

            if alpha_creation != alpha_annihilation:
                return False

            if beta_creation != beta_annihilation:
                return False

        return True

    def is_two_body_number_conserving(self) -> bool:
        """Check whether the operator is at most two-body and
        conserves particle number."""
        for term in self:

            n_operators = len(term)

            # allowed:
            # 0 -> identity
            # 2 -> one-body
            # 4 -> two-body
            if n_operators not in (0, 2, 4):
                return False

        return self.conserves_particle_number()
