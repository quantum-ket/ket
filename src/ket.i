%include <std_string.i>
%include <std_vector.i>
%include <stdint.i>

%template(vec_str) std::vector<std::string>;

%module ket
%{
    #include "include/pyket.hpp"
    #include "libket/include/ket"
%}

%exception {
  try {
    $action
  } catch(std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "include/pyket.hpp"
%include "libket/include/ket"

%pythoncode 
%{
from sys import argv
pyket___ = PyKet(argv)
%}