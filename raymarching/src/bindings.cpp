#include <torch/extension.h>

#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.def("composite_rays_train_forward", &composite_rays_train_forward, "composite_rays_train_forward (CUDA)");
    m.def("composite_rays_train_backward", &composite_rays_train_backward, "composite_rays_train_backward (CUDA)");
    m.def("march_rays", &march_rays, "march rays (CUDA)");
    m.def("composite_rays", &composite_rays, "composite rays (CUDA)");
    m.def("compact_rays", &compact_rays, "compact rays (CUDA)");
}