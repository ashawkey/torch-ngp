#ifndef _RAYMARCHING_H
#define _RAYMARCHING_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

// _backend.generate_points(rays_o, rays_d, density_grid, bound, N, H, M, points, offsets)
void generate_points(at::Tensor rays_o, at::Tensor rays_d, at::Tensor grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, at::Tensor points, at::Tensor rays,  at::Tensor counter);
void generate_points_cuda(const float *rays_o, const float *rays_d, const float *grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, float *points, int *rays, int *counter);


// _backend.accumulate_rays_forward(sigmas, rgbs, rays, bound, M, N, depth, image)
void accumulate_rays_forward(at::Tensor sigmas, at::Tensor rgbs, at::Tensor points, at::Tensor rays, const float bound, const uint32_t M, const uint32_t N, at::Tensor depth, at::Tensor image);
void accumulate_rays_forward_cuda(const float *sigmas, const float *rgbs, const float *points, const int *rays, const float bound, const uint32_t M, const uint32_t N, float *depth, float *image);

void accumulate_rays_backward(at::Tensor grad, at::Tensor sigmas, at::Tensor rgbs, at::Tensor points, at::Tensor rays, at::Tensor image, const float bound, const uint32_t M, const uint32_t N, at::Tensor grad_sigmas, at::Tensor grad_rgbs);
void accumulate_rays_backward_cuda(const float *grad, const float *sigmas, const float *rgbs, const float *points, const int *rays, const float *image, const float bound, const uint32_t M, const uint32_t N, float *grad_sigmas, float *grad_rgbs);
#endif