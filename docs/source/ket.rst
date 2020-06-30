Ket quantum programming language
================================

This documentation aims to present the Ket quantum programming embedded in
Python. Previous knowledge of Python and quantum computation is required.

The type quant 
--------------

The type ``quant`` holds an array of qubits and it is initialized with 
``quant(size)`` or ``quant.dirty(size)``.  To reference a single qubit of a
quant use brackets.

.. code-block:: python

    a = quant(3)        # a = |000>
    b = quant,dirty(2)  # b = 2 qubits quant in a random state
    c = a[0]            # c is a reference to the first qubit of a
    d = b[1]            # d is a reference to the last qubit of a

Quantum Gates
^^^^^^^^^^^^^

The available quantum gate are:

+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gate name    | Function                       | Matrix representation                                                                                                                                             |
+==============+================================+===================================================================================================================================================================+
| Pauli X      | ``x(q)``                       | :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`                                                                                                              |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Pauli Y      | ``y(q)``                       | :math:`\begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}`                                                                                                             |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Pauli Z      | ``z(q)``                       | :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}`                                                                                                             |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Hadamard     | ``h(q)``                       | :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}`                                                                                           |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Phase        | ``s(q)``                       | :math:`\begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}`                                                                                                              |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Phase dagger | ``sd(q)``                      | :math:`\begin{bmatrix} 1 & 0 \\ 0 & -i \end{bmatrix}`                                                                                                             |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| T            | ``t(q)``                       | :math:`\begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}`                                                                                                     |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| T dagger     | ``td(q)``                      | :math:`\begin{bmatrix} 1 & 0 \\ 0 & (e^{i\pi/4})^\dagger \end{bmatrix}`                                                                                           |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| U1           | ``u1(_lambda, q)``             | :math:`\begin{bmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{bmatrix}`                                                                                                   |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| U2           | ``u2(phi, _lambda, q)``        | :math:`\frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -e^{i\lambda} \\ e^{i\phi} & e^{i(\lambda+\phi)} \end{bmatrix}`                                                     |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| U3           | ``u3(theta, phi, _lambda, q)`` | :math:`\begin{bmatrix} \cos{\theta\over2} & -e^{i\lambda}\sin{\theta\over2} \\ e^{i\phi}\sin{\theta\over2} & e^{i(\lambda+\phi)}\cos{\theta\over2} \end{bmatrix}` |
+--------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+


.. code-block:: python

    a = quant(5)                         # a = |00000>
    x(a) # apply Pauli X on every qubit of a = |11111>


Controlled operations
^^^^^^^^^^^^^^^^^^^^^

To apply a controlled quantum operation use the statement ``with control(c):``
or the function ``ctrl(c, gate, *args)``.

For example, to apply a CNOT or a Toffoli gate:

.. code-block:: python

    c = quant(2)
    t = quant(1)

    # CNOT(c[0], t)
    with control(c[0]):
        x(t)            

    # CNOT(c[0], t)
    ctrl(c[0], x, t)    

    # Toffoli(c[0], c[1], t)
    with control(c):
        x(t)            

    # Toffoli(c[0], c[1], t)
    ctrl(c, x, t)    

.. warning:: ``with control(c):`` and ``ctrl(c, gate, *args)`` does not operate
    with ``measure(q)``, ``qalloc(n)``, or ``qalloc_dirty(n)``.
    
Inverse operations
^^^^^^^^^^^^^^^^^^

To apply a inverse quantum operation use the statement ``with inverse():`` or
the function ``adj(gate, *args)``.

For example to apply a inverse Quantum Fourier Transform:

.. code-block:: python
    
    # Quantum Fourier Transform
    def qft(q):
        w = lambda k : pi*k/2
        for i in range(len(q)):
            for j in range(i):
                ctrl(q[i], u1, w(i-j), q[j])
            h(q[i])
    
    q = quant(5)

    # inverse Quantum Fourier Transform 
    with inverse():
        qft(q)
        
    # inverse Quantum Fourier Transform 
    adj(qft, q)
        
.. warning:: ``with inverse():`` and ``adj(gate, *args)`` does not operate with
    ``measure(q)``, ``qalloc(n)``, or ``qalloc_dirty(n)``.

The type future 
---------------

The type ``future`` holds an ``int`` that is primarily available at the quantum
computer, as proposed by [arXiv:2006.00131]_.
Its central usage is to reference measurement results, but it also stores the
result of operations with measurement results and ``int``.  
To receive the value of a ``future`` use the function ``.get()``, which will
execute the necessary quantum code.

.. code-block:: python

    q = qalloc(60)
    h(q)

    m = measure(q) # m is a future 
    m5 = m * 5     # m5 is a future

    result = m5.get() # result is a int
    
.. note:: The available operations between ``future``-``future`` and 
    ``future``-``int`` are ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``+``,
    ``-``, ``*``, ``/``, ``<<``, ``>>``, ``and``, ``xor``, and ``or``.

Statement integration 
^^^^^^^^^^^^^^^^^^^^^
