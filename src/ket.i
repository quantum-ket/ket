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
label.__repr__ = lambda self : '<Ket label; '+self.this.__repr__()+'>'
%}
