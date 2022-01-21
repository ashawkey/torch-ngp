#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [B, L * C], float
// H: base resolution
void encode_forward(at::Tensor inputs, at::Tensor embeddings, at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx);
void encode_backward(at::Tensor grad, at::Tensor inputs, at::Tensor embeddings, at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx, at::Tensor grad_inputs);
void encode_forward_cuda(const float *inputs, const float *embeddings, const int *offsets, float *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, float *dy_dx);
void encode_backward_cuda(const float *grad, const float *inputs, const float *embeddings, const int *offsets, float *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, float *dy_dx, float *grad_inputs);
#endif