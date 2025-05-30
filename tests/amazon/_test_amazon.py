# from functools import partial
# from math import pi, sqrt
# from plotly import express as px

from ket import *
from ket import ket_version
from ket.amazon import AmazonBraket

print(ket_version())

print("Iniciando criação do AmazonBraket...")
try:
    braket = AmazonBraket(num_qubits=2)
    print("AmazonBraket criado com sucesso.")
except Exception as e:
    print(f"Erro ao criar AmazonBraket: {e}")

print("Iniciando braket.connect()...")
try:
    ket_configuration = braket.connect()
    print("braket.connect() executado com sucesso.")
    # Se travar aqui, o problema está no connect(),
    # provavelmente na chamada a API["ket_make_configuration"].
except Exception as e:
    print(f"Erro ao executar braket.connect(): {e}")