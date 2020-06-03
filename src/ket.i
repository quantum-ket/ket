%include <std_string.i>
%include <std_vector.i>

%template(vec_str) std::vector<std::string>;

%module ket
%{
    #include "include/pyket.hpp"
%}

%exception {
  try {
    $action
  } catch(std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "include/pyket.hpp"

%pythoncode 
%{
from sys import argv
___pyket = ___PyKet(argv)
%}