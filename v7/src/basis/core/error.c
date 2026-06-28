
#include "basis/core/error.h"
__thread basis_error_t _basis_last_error = BASIS_SUCCESS;
__thread const char* _basis_last_error_msg = "";
__thread bool _basis_error_quiet = false;
