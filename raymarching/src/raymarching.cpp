#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include "raymarching.h"
#include "utils.h"

void generate_points(at::Tensor rays_o, at::Tensor rays_d, at::Tensor grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, at::Tensor points, at::Tensor rays, at::Tensor counter) {
    CHECK_CUDA(rays_o);
    CHECK_CUDA(rays_d);
    CHECK_CUDA(grid);
    CHECK_CUDA(points);
    CHECK_CUDA(rays);
    CHECK_CUDA(counter);

    CHECK_CONTIGUOUS(rays_o);
    CHECK_CONTIGUOUS(rays_d);
    CHECK_CONTIGUOUS(grid);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(rays);
    CHECK_CONTIGUOUS(counter);

    CHECK_IS_FLOAT(rays_o);
    CHECK_IS_FLOAT(rays_d);
    CHECK_IS_FLOAT(grid);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(rays);
    CHECK_IS_INT(counter);

    generate_points_cuda(rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), grid.data_ptr<float>(), mean_density, iter_density, bound, N, H, M, points.data_ptr<float>(), rays.data_ptr<int>(), counter.data_ptr<int>());
}


void accumulate_rays_forward(at::Tensor sigmas, at::Tensor rgbs, at::Tensor points, at::Tensor rays, const float bound, const uint32_t M, const uint32_t N, at::Tensor depth, at::Tensor image) {

    CHECK_CUDA(sigmas);
    CHECK_CUDA(rgbs);
    CHECK_CUDA(points);
    CHECK_CUDA(rays);
    CHECK_CUDA(depth);
    CHECK_CUDA(image);

    CHECK_CONTIGUOUS(sigmas);
    CHECK_CONTIGUOUS(rgbs);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(rays);
    CHECK_CONTIGUOUS(depth);
    CHECK_CONTIGUOUS(image);

    CHECK_IS_FLOAT(sigmas);
    CHECK_IS_FLOAT(rgbs);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(rays);
    CHECK_IS_FLOAT(depth);
    CHECK_IS_FLOAT(image);

    accumulate_rays_forward_cuda(sigmas.data_ptr<float>(), rgbs.data_ptr<float>(), points.data_ptr<float>(), rays.data_ptr<int>(), bound, M, N, depth.data_ptr<float>(), image.data_ptr<float>());
}


void accumulate_rays_backward(at::Tensor grad, at::Tensor sigmas, at::Tensor rgbs, at::Tensor points, at::Tensor rays, at::Tensor image, const float bound, const uint32_t M, const uint32_t N, at::Tensor grad_sigmas, at::Tensor grad_rgbs) {

    CHECK_CUDA(grad);
    CHECK_CUDA(sigmas);
    CHECK_CUDA(rgbs);
    CHECK_CUDA(points);
    CHECK_CUDA(rays);
    CHECK_CUDA(image);
    CHECK_CUDA(grad_sigmas);
    CHECK_CUDA(grad_rgbs);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(sigmas);
    CHECK_CONTIGUOUS(rgbs);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(rays);
    CHECK_CONTIGUOUS(image);
    CHECK_CONTIGUOUS(grad_sigmas);
    CHECK_CONTIGUOUS(grad_rgbs);

    CHECK_IS_FLOAT(grad);
    CHECK_IS_FLOAT(sigmas);
    CHECK_IS_FLOAT(rgbs);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(rays);
    CHECK_IS_FLOAT(image);
    CHECK_IS_FLOAT(grad_sigmas);
    CHECK_IS_FLOAT(grad_rgbs);

    accumulate_rays_backward_cuda(grad.data_ptr<float>(), sigmas.data_ptr<float>(), rgbs.data_ptr<float>(), points.data_ptr<float>(), rays.data_ptr<int>(), image.data_ptr<float>(), bound, M, N, grad_sigmas.data_ptr<float>(), grad_rgbs.data_ptr<float>());
}
