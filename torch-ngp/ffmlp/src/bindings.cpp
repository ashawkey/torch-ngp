#include <torch/extension.h>

#include "ffmlp.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ffmlp_forward", &ffmlp_forward, "ffmlp_forward (CUDA)");
    m.def("ffmlp_inference", &ffmlp_inference, "ffmlp_inference (CUDA)");
    m.def("ffmlp_backward", &ffmlp_backward, "ffmlp_backward (CUDA)");
    m.def("allocate_splitk", &allocate_splitk, "allocate_splitk (CUDA)");
    m.def("free_splitk", &free_splitk, "free_splitk (CUDA)");
}