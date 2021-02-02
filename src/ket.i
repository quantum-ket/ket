/* MIT License
 * 
 * Copyright (c) 2020 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
 * Copyright (c) 2020 Rafael de Santiago <r.santiago@ufsc.br>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

%include <std_string.i>
%include <std_vector.i>
%include <stdint.i>

%template(vec_float) std::vector<double>;

%module ket
%{
    #include "libket/include/ket"
%}

%exception {
  try {
    $action
  } catch(std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%extend ket::quant {
    //! x.__getitem__(y) <==> x[y]
    quant __getitem__(PyObject *param) {
        if (PySlice_Check(param)) {
            Py_ssize_t start, stop, step, size = $self->len();
            PySlice_Unpack(param, &start, &stop, &step);
            PySlice_AdjustIndices(size, &start, &stop, step);
            return (*$self)(start, stop, step);
        } else if (PyLong_Check(param)) {
            return (*$self)(PyLong_AsLong(param));
        }
        throw std::runtime_error("quant.__getitem__ accepts a slice or int as param");
    }
}

%typemap(out) std::vector<unsigned long long> ket::dump::get_states
%{
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); i++) {
    PyList_SetItem($result, i, PyLong_FromUnsignedLongLong($1.at(i)));
  }
%}

%typemap(out) std::vector<std::complex<double>> ket::dump::amplitude
%{
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); i++) {
    PyList_SetItem($result, i, PyComplex_FromDoubles($1.at(i).real(), $1.at(i).imag()));
  }
%}

%include "libket/include/ket"

%pythoncode 
%{

X  = x
Y  = y
Z  = z
H  = h
S  = s
SD = sd
T  = t
TD = td    
U1 = u1
p  = u1
P  = p
U2 = u2
U3 = u3
RX = rx
RY = ry
RZ = rz

class __quant__iter__:
    def __init__(self, q):
        self.q = q
        self.idx = -1
        self.size = len(q)

    def __next__(self): 
        self.idx += 1
        if self.idx < self.size:
            return self.q[self.idx]
        raise StopIteration

quant.__iter__ = lambda self : __quant__iter__(self)

quant.__enter__ = lambda self : self

def __quant__exit__(self, type, value, tb):
    if not self.is_free():
        raise RuntimeError('non-free quant at the end of scope')

quant.__exit__ = __quant__exit__

quant.__repr__ = lambda self : '<Ket quant; '+str(len(self))+' qubits; '+self.this.__repr__()+'>'
future.__repr__ = lambda self : '<Ket future; '+self.this.__repr__()+'>'
dump.__repr__ = lambda self : '<Ket dump; '+self.this.__repr__()+'>'
metrics.__repr__ = lambda self : '<Ket metrics; '+self.this.__repr__()+'>'
label.__repr__ = lambda self : '<Ket label; '+self.this.__repr__()+'>'

class run:
    """Run the quantum operations in a new process.
    
    Usage:

    .. code-block:: ket

        with run():
            ...

    """
    def __enter__ (self):
        process_begin()
    
    def __exit__ (self, type, value, tb):
        process_end()

class inverse:
    """Apply the quantum operations backwards.
    
    Usage:

    .. code-block:: ket

        with inverse():
            ...
            
    """
    def __enter__ (self):
        adj_begin()     

    def __exit__ (self, type, value, tb):
        adj_end()     

class control:
    """Apply controlled quantum operations.
    
    Usage:

    .. code-block:: ket
    
        with control(cont):
            ...
            
    """
    def __init__(self, c : quant):
        self.c = c

    def __enter__ (self):
        ctrl_begin(self.c)
     
    def __exit__ (self, type, value, tb):
        ctrl_end()

def ctrl(control : quant, func, *args, **kwargs):
    """Add qubits of controll to a operation call."""
    ctrl_begin(control)
    ret = func(*args, **kwargs)
    ctrl_end()
    return ret

def adj(func, *args, **kwargs):
    """Call the inverse of a quantum operation."""
    adj_begin()
    func(*args, **kwargs)
    adj_end()

%}
