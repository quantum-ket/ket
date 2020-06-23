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

.. image:: https://snapcraft.io/static/images/badges/en/snap-store-black.svg
   :target: https://snapcraft.io/ket
   :alt: Get it from the Snap Store

.. code-block:: bash

   sudo snap install ket --edge
   
For help to install snapd see https://snapcraft.io/docs/installing-snapd

Contents
--------

.. toctree::
   :maxdepth: 2

   ket
   references 
   license
