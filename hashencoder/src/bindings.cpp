#include <torch/extension.h>

#include "hashencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_forward", &encode_forward, "encode forward (CUDA)");
    m.def("encode_backward", &encode_backward, "encode backward (CUDA)");
}