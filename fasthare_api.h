#pragma once
#include <string>
#include "fasthare.h"   // 来自 fasthare sdist 的 src/fasthare.h

namespace fasthare_api {

using ski    = FastHare::ski;              // vector<tuple<int,int,double>>
using output = FastHare::fasthare_output;  // tuple<ski, vector<int>, vector<int>, double>

// 等价于 Python: fasthare.fasthare_reduction(...)
output fasthare_reduction(
    const ski& sk_ising = ski{},
    const std::string& file = "",
    double alpha = 1.0,
    const std::string& log_file = ""
);

} // namespace fasthare_api
