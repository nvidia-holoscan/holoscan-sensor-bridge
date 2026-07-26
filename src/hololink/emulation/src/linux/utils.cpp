#include "../../utils.hpp"
#include <stdexcept>

extern "C" void Error_Handler(const char* str)
{
    throw std::runtime_error(str);
}
