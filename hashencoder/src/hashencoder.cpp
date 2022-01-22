#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include "hashencoder.h"
#include "utils.h"

void hash_encode_forward(at::Tensor inputs, at::Tensor embeddings, at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(outputs);
    CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOAT(inputs);
    CHECK_IS_FLOAT(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOAT(outputs);
    CHECK_IS_FLOAT(dy_dx);

    hash_encode_forward_cuda(inputs.data_ptr<float>(), embeddings.data_ptr<float>(), offsets.data_ptr<int>(), outputs.data_ptr<float>(), B, D, C, L, H, calc_grad_inputs, dy_dx.data_ptr<float>());
}

void hash_encode_backward(at::Tensor grad, at::Tensor inputs, at::Tensor embeddings, at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx, at::Tensor grad_inputs) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_inputs);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOAT(grad);
    CHECK_IS_FLOAT(inputs);
    CHECK_IS_FLOAT(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOAT(grad_embeddings);
    CHECK_IS_FLOAT(dy_dx);
    CHECK_IS_FLOAT(grad_inputs);
    
    hash_encode_backward_cuda(grad.data_ptr<float>(), inputs.data_ptr<float>(), embeddings.data_ptr<float>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<float>(), B, D, C, L, H, calc_grad_inputs, dy_dx.data_ptr<float>(), grad_inputs.data_ptr<float>());
}