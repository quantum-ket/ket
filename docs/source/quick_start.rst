Quick Start
===========

Install
-------

The easy way to start programming in Ket is to install the following Snap
packages. For instruction on how to install and enable the Snap daemon on your
linux distribution see https://snapcraft.io/docs/installing-snapd.

.. code-block:: bash

   sudo snap install kbw --edge
   sudo snap install ket --edge

See https://gitlab.com/quantum-ket/ket for other installation methods.

Random Number Generator
-----------------------

.. code-block:: ket

   def random(n_bits):
     with run():
        q = quant(n_bits)
        h(q)
        return measure(q).get()

   n_bits = 32
   print(n_bits, 'bits random number:', random(n_bits))
..

   32 bits random number: 4155389359


 
