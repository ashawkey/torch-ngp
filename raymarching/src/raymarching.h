#pragma once

#include <stdint.h>
#include <torch/torch.h>


void near_far_from_aabb(at::Tensor rays_o, at::Tensor rays_d, at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars);

void march_rays_train(at::Tensor rays_o, at::Tensor rays_d, at::Tensor grid, const float mean_density, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, at::Tensor nears, at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, const uint32_t perturb);
void composite_rays_train_forward(at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor rays, const uint32_t M, const uint32_t N, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);
void composite_rays_train_backward(at::Tensor grad_weights_sum, at::Tensor grad_image, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor rays, at::Tensor weights_sum, at::Tensor image, const uint32_t M, const uint32_t N, at::Tensor grad_sigmas, at::Tensor grad_rgbs);

void march_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor rays_o, at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, at::Tensor density_grid, const float mean_density, at::Tensor nears, at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, const uint32_t perturb);
void composite_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);
void compact_rays(const uint32_t n_alive, at::Tensor rays_alive, at::Tensor rays_alive_old, at::Tensor rays_t, at::Tensor rays_t_old, at::Tensor alive_counter);