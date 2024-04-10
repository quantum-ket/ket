from ket import *

p = Process(execution="batch", optimize=True)


a, b = p.alloc(2)

CNOT(H(a), b)  # Bell state preparation

H(a)
m1 = sample(a+b, 1024)


X(a)
X(b)

m3 = measure(a)
m4 = measure(b)


print(p.get_qasmv2())

p.execute()

print(m1.value)
print(m3.value)
print(m4.value)
