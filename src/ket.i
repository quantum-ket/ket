/* Copyright 2020, 2021 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
 * Copyright 2020, 2021 Rafael de Santiago <r.santiago@ufsc.br>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

%typemap(out) std::vector<std::vector<unsigned long>> ket::dump::get_states
%{
    $result = PyList_New($1.size());
    for (size_t i = 0; i < $1.size(); i++) {
        auto& state = $1.at(i);
        auto* py_int = PyLong_FromUnsignedLong(state.back());
        for (auto it = state.rbegin()+1; it != state.rend(); ++it) {
            py_int = PyNumber_Lshift(py_int, PyLong_FromLong(64));
            py_int = PyNumber_Or(py_int, PyLong_FromUnsignedLong(*it));
        }
        std::cout << std::endl;
        PyList_SetItem($result, i, py_int);
    }
%}

%typemap(out) std::vector<std::complex<double>> ket::dump::amplitude
%{
    $result = PyList_New($1.size());
    for (size_t i = 0; i < $1.size(); i++) {
        PyList_SetItem($result, i, PyComplex_FromDoubles($1.at(i).real(), $1.at(i).imag()));
    }
%}

%typemap(in) (std::vector<unsigned long> idx) 
%{  
    $1 = std::vector<unsigned long>();
    $1.push_back(PyLong_AsUnsignedLongMask($input));
    for (int i = 0; i < arg1->nbits/64; i++) {
        $input = PyNumber_Rshift($input, PyLong_FromLong(64));
        $1.push_back(PyLong_AsUnsignedLongMask($input));
    } 
%}


%include "libket/include/ket"

%pythoncode 
%{
label.__repr__ = lambda self : '<Ket label; '+self.this.__repr__()+'>'
%}
