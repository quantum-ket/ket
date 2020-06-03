#include "../include/pyket.hpp"
#include <vector>
#include <string>

inline char* str_copy(const std::string& str) {
    char* ptr = new char[str.size()+1];
    std::copy(str.begin(), str.end(), ptr);
    ptr[str.size()] = '\0';
    return ptr;
}

___PyKet::___PyKet(const std::vector<std::string> &argv) {
    int argc = argv.size();
    char* c_argv[argc];
    for (int i = 0; i < argc; i++) 
        c_argv[i] = str_copy(argv[i]);
    ket_init_new(argc, c_argv);    
}

___PyKet::~___PyKet() {
    ket_init_free();
}