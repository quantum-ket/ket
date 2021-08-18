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
%include <std_complex.i>
%include <stdint.i>

%template(vec_float) std::vector<double>;
%template(vec_uint) std::vector<unsigned long>;
%template(vec_vec_uint) std::vector<std::vector<unsigned long>>;
%template(vec_complex) std::vector<std::complex<double>>;
%template(complex_d) std::complex<double>;


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

%include "libket/include/ket"

%pythoncode 
%{
label.__repr__ = lambda self : '<Ket label; '+self.this.__repr__()+'>'
%}
