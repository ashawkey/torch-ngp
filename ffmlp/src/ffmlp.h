#pragma once

#include <stdint.h>
#include <torch/torch.h>


// activation: should have been enum, here we just use int.
void ffmlp_forward(const at::Tensor inputs, const at::Tensor weights, const uint32_t B, const uint32_t input_dim, const uint32_t output_dim, const uint32_t hidden_dim, const uint32_t num_layers, const uint32_t activation_, const uint32_t output_activation_, at::Tensor forward_buffer, at::Tensor outputs);
void ffmlp_inference(const at::Tensor inputs, const at::Tensor weights, const uint32_t B, const uint32_t input_dim, const uint32_t output_dim, const uint32_t hidden_dim, const uint32_t num_layers, const uint32_t activation_, const uint32_t output_activation_, at::Tensor inference_buffer, at::Tensor outputs);

void ffmlp_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor weights, const at::Tensor forward_buffer, const uint32_t B, const uint32_t input_dim, const uint32_t output_dim, const uint32_t hidden_dim, const uint32_t num_layers, const uint32_t activation, const uint32_t output_activation, const bool calc_grad_inputs, at::Tensor backward_buffer, at::Tensor grad_inputs, at::Tensor grad_weights);

void allocate_splitk(size_t size);
void free_splitk();