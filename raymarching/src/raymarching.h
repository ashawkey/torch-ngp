#pragma once

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

// _backend.generate_points(rays_o, rays_d, density_grid, bound, N, H, M, points, offsets)
void generate_points(at::Tensor rays_o, at::Tensor rays_d, at::Tensor grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, at::Tensor points, at::Tensor rays,  at::Tensor counter);

// _backend.accumulate_rays_forward(sigmas, rgbs, rays, bound, M, N, depth, image)
void accumulate_rays_forward(at::Tensor sigmas, at::Tensor rgbs, at::Tensor points, at::Tensor rays, const float bound, const uint32_t M, const uint32_t N, at::Tensor depth, at::Tensor image);
void accumulate_rays_backward(at::Tensor grad, at::Tensor sigmas, at::Tensor rgbs, at::Tensor points, at::Tensor rays, at::Tensor image, const float bound, const uint32_t M, const uint32_t N, at::Tensor grad_sigmas, at::Tensor grad_rgbs);
