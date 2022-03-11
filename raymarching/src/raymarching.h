#pragma once

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

// _backend.march_rays_train(rays_o, rays_d, density_grid, bound, N, H, M, points, offsets)
void march_rays_train(at::Tensor rays_o, at::Tensor rays_d, at::Tensor grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, const uint32_t perturb);

// _backend.composite_rays_train_forward(sigmas, rgbs, rays, bound, M, N, weights_sum, image)
void composite_rays_train_forward(at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor rays, const float bound, const uint32_t M, const uint32_t N, at::Tensor weights_sum, at::Tensor image);
void composite_rays_train_backward(at::Tensor grad_weights_sum, at::Tensor grad, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor rays, at::Tensor weights_sum, at::Tensor image, const float bound, const uint32_t M, const uint32_t N, at::Tensor grad_sigmas, at::Tensor grad_rgbs);

// inference functions
void march_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor rays_o, at::Tensor rays_d, const float bound, const uint32_t H, at::Tensor density_grid, const float mean_density, at::Tensor near, at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, const uint32_t perturb);
void composite_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image);
void compact_rays(const uint32_t n_alive, at::Tensor rays_alive, at::Tensor rays_alive_old, at::Tensor rays_t, at::Tensor rays_t_old, at::Tensor alive_counter);