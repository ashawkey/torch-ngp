#include <torch/extension.h>

#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_points", &generate_points, "generate_points (CUDA)");
    m.def("accumulate_rays_forward", &accumulate_rays_forward, "accumulate_rays_forward (CUDA)");
    m.def("accumulate_rays_backward", &accumulate_rays_backward, "accumulate_rays_backward (CUDA)");
}