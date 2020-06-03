#pragma once
#include "../libket/include/ket"
#include <vector>
#include <string>

class PyKet {
public:
    PyKet(const std::vector<std::string> &argv);
    ~PyKet();
};