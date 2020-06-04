%include <std_string.i>
%include <std_vector.i>
%include <stdint.i>

%template(vec_str) std::vector<std::string>;

%module __init__
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
ket___ = PyKet(argv)

def ctrl(control, func, *args):
    ctrl_begin(control)
    ret = func(*args)
    ctrl_end(control)
    return ret

def adj(func, *args):
    adj_begin()
    func(*args)
    adj_end()

%}