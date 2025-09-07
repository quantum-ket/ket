# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Bell state preparation."""
from ket import *
from ket.amazon import AmazonBraket
from math import sqrt

device = AmazonBraket()                                                                    # LocalSimulator
# device = AmazonBraket('arn:aws:braket:::device/quantum-simulator/amazon/tn1')            # TN1
# device = AmazonBraket('arn:aws:braket:::device/quantum-simulator/amazon/dm1')            # DM1
# device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1')                # IonQ Aria-1
# device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/ionq/Aria-2')                # IonQ Aria-2
# device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1')               # IonQ Forte-1
# device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1')    # IonQ Forte-Enterprise-1
# device = AmazonBraket('arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet')                # IQM Garnet
# device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/quera/Aquila')               # QuEra Aquila
# device = AmazonBraket('arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3')            # Rigetti Ankaa-3



process = Process(device)
a, b = process.alloc(2)

X(a + b)
CNOT(H(a), b)

with obs():
    a0 = Z(a)
    a1 = X(a)
    b0 = -(X(b) + Z(b)) / sqrt(2)
    b1 = (X(b) - Z(b)) / sqrt(2)
    h = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1

print(exp_value(h).get())
