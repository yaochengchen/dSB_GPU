#include "fasthare_api.h"

namespace fasthare_api {

output fasthare_reduction(
    const ski& sk_ising,
    const std::string& file,
    double alpha,
    const std::string& log_file
) {
    if (!file.empty()) {
        FastHare fh(file, alpha);
        if (!log_file.empty()) fh.set_log(log_file);
        fh.run();
        return fh.get_output();
    } else {
        FastHare fh(sk_ising, alpha);
        if (!log_file.empty()) fh.set_log(log_file);
        fh.run();
        return fh.get_output();
    }
}

} // namespace fasthare_api
