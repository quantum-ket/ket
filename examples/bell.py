# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Bell state preparation."""
import ket


p = ket.Process()
a = p.alloc()
b = p.alloc()

ket.H(a)
ket.ctrl(a, ket.X)(b)

print(ket.dump(a + b).show())
print(ket.sample(a + b).value)
print(ket.exp_value(ket.Pauli("X", a + b)).value)
print(ket.exp_value(ket.Pauli("Y", a + b)).value)
print(ket.exp_value(ket.Pauli("Z", a + b)).value)
print(ket.measure(a + b).value)
