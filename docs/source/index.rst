Welcome to Ket's documentation!
===============================

Ket is a quantum programming framework that features:

* A Python-embedded quantum programming language with the same name,
* The libket shred library for C++, C, and Python, based in runtime
  architecture present in [arXiv:2006.00131]_,
* The Ket Bitwise Simulator for quantum circuit simulator with bitwise
  operation [arXiv:2004.03560]_.

Source code
-----------

* Ket interpreter (keti) and libket for Python: https://gitlab.com/quantum-ket/ket
* libket for C++ and C: https://gitlab.com/quantum-ket/libket
* Ket Bitwise Simulator (kbw): https://gitlab.com/quantum-ket/kbw 

Install ket on your Linux distribution
--------------------------------------

.. code-block:: bash

   sudo snap install ket --beta
   
For help to install snapd see https://snapcraft.io/docs/installing-snapd

Contents
--------

.. toctree::
   :maxdepth: 2

   ket 
   license

References
==========
.. [arXiv:2006.00131] `Classical and Quantum Data Interaction in Programming Languages: A Runtime Architecture`__
   Evandro Chagas Ribeiro da Rosa, Rafael de Santiago
__ https://arxiv.org/abs/2006.00131

.. [arXiv:2004.03560] `QSystem: bitwise representation for quantum circuit simulations`__
   Evandro Chagas Ribeiro da Rosa, Bruno G. Taketani
__ https://arxiv.org/abs/2004.03560


