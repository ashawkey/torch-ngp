#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>
#include <vector>

#include <mma.h>

#include "utils.h"
#include "cutlass_matmul.h"


__host__ __device__ Activation convert_activation(const uint32_t activation) {
    switch (activation) {
        case 0: return Activation::ReLU;
        case 1: return Activation::Exponential;
        case 2: return Activation::Sine;
        case 3: return Activation::Sigmoid;
        case 4: return Activation::Squareplus;
        case 5: return Activation::Softplus;
        case 6: return Activation::None;
        default: return Activation::None;
    }
}

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

void check_shmem_error(cudaError_t error) {
	if (error != cudaSuccess) {
		throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce `n_neurons` or use `CutlassMLP` (better compatibility but slower) instead."};
	}
}


template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, bool BACKWARD=false>
__device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
	//           Can be forward activations or backward activations, depending on caller.
	// weights_this_layer points to the weight matrix of the current layer.
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// activation_aux points to additional arguments that the activation function may depend on. Points to the hidden forward activations when computing backward activations.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// If we're performing the backward pass, weights must be loaded in transposed form, which
	// is achieved by interpreting the memory in row_major instead of col_major order.
	using weights_layout_t = std::conditional_t<BACKWARD, wmma::row_major, wmma::col_major>;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, weights_layout_t> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__syncthreads();

	// Load N_BLOCKS chunks of weights from global memory into registers.
	#pragma unroll
	for (uint32_t i = 0; i < N_BLOCKS; ++i) {
		if (BACKWARD) {
			// If we're performing the backward pass, additional index swizzling is needed to
			// load the weights in transposed form.
			wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i * WIDTH + weights_col, WIDTH);
		} else {
			wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i + weights_col * WIDTH, WIDTH);
		}
	}

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::fill_fragment(result_frag[l], 0.0f);

		#pragma unroll
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), WIDTH + SKEW);
			wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
		}

		// Activation
		if (BACKWARD) {
			// Load the temporary forward matrix for the relu transfer
			wmma::load_matrix_sync(act_frag, activation_aux + weights_col + (threadIdx.z + l * BLOCK_DIM_Z) * 16 * WIDTH, WIDTH);
			warp_activation_backward<__half>(activation, result_frag[l], act_frag, result_frag[l]);
		} else {
			warp_activation<__half>(activation, result_frag[l], result_frag[l]);
		}
	}

	__syncthreads();

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::store_matrix_sync(act_shmem + weights_col + (threadIdx.z + l * BLOCK_DIM_Z) * 16 * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
		}
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock) {
	// act_shmem will be filled by the thread block's chunk of input_threadblock

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	#pragma unroll
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)] = *(int4*)&input_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH];
	}
}


template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const uint32_t in_width) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// input_threadblock points to the thread block's chunk of the input batch in global memory
	// weights_this_layer points to the weight matrix of the current layer
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// in_width is the dynamic width of the input layer

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t INPUT_SKEW = 8;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__half* __restrict__ weights_shmem = act_shmem + BLOCK_DIM_Z * 16 * (in_width + INPUT_SKEW);

	// Load input weight matrix (fits completely into shared memory)
	// Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS*BLOCK_DIM_Z warps
	const uint32_t n_elems_per_load = N_BLOCKS * 32 * BLOCK_DIM_Z * 8;
	const uint32_t thread_elem_idx = (li + wi * 32 + threadIdx.z * N_BLOCKS * 32) * 8;

	const uint32_t n_elems_b = WIDTH * in_width;

	#pragma unroll
	for (uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
		const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
		*(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];
	}

	const uint32_t n_tensor_ops = in_width / 16;

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		// Load chunk of inputs into shmem.
		// This is faster than loading it from gmem directly, even though it is only used once.
		// (Possibly due to latency hiding through staging.)
		const uint32_t n_elems_a = BLOCK_DIM_Z * 16 * in_width;

		#pragma unroll
		for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
			const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
			*(int4*)&act_shmem[idx_skewed] = *(int4*)&input_threadblock[l * n_elems_a + idx];
		}

		__syncthreads();

		wmma::fill_fragment(result_frag[l], 0.0f);
		#pragma unroll
		for (uint32_t i = 0; i < n_tensor_ops; ++i) {
			// Load chunk of inputs and weights from shared memory and multiply them
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * threadIdx.z) * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
			wmma::load_matrix_sync(weights_frag, weights_shmem + 16 * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
			wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
		}

		__syncthreads();

		warp_activation<__half>(activation, result_frag[l], result_frag[l]);
	}

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::store_matrix_sync(act_shmem + weights_col + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		#pragma unroll
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
		}
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__device__ void threadblock_last_layer_forward(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out, const uint32_t batch_size, const nvcuda::wmma::layout_t output_layout) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// weights_this_layer points to the weight matrix of the current layer
	// out points to the location where the result produced by the thread block should be written to.
	//   Can be nullptr if nothing should be written.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	__half* __restrict__ weights_shmem = act_shmem + N_ITERS * BLOCK_DIM_Z * 16 * (WIDTH + SKEW);

	const uint32_t weights_row = (8 * li) % WIDTH;
	const uint32_t weights_col = (8 * li + 8 * 32 * wi) / WIDTH;

	// Load weight matrix into shared memory for the last multiplication.
	// Loading into shared memory as opposed to directly into registers is faster
	// because unlike in the previous layers, each warp uses the same entries of the weight matrix.
	if (threadIdx.z == 0) {
		*(int4*)&weights_shmem[weights_row + weights_col * (WIDTH + SKEW)] = *(int4*)&weights_this_layer[weights_row + weights_col * WIDTH];
		//printf("[last forward] base=%d, shmem=%d, weight=%d\n", N_ITERS * BLOCK_DIM_Z * 16 * (WIDTH + SKEW), weights_row + weights_col * (WIDTH + SKEW), weights_row + weights_col * WIDTH);
	}

	__syncthreads();

	#pragma unroll
	for (uint32_t i = 0; i < N_BLOCKS; ++i)
		wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16 * i, WIDTH + SKEW);

	// Perform last layer by parallelizing over iters
	for (uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
		wmma::fill_fragment(result_frag, 0.0f);

		#pragma unroll
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of the weight matrix
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * (threadIdx.z + idx * BLOCK_DIM_Z)) * (WIDTH + SKEW), WIDTH + SKEW);
			wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
		}

		warp_activation<__half>(activation, result_frag, result_frag);

		if (output_layout == wmma::mem_row_major) {
			wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16 * 16, result_frag, 16, output_layout);
			//printf("[last forward] RM write out %d, batch %d\n", (threadIdx.z + idx * BLOCK_DIM_Z) * 16 * 16, 16);
		} else {
			wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16, result_frag, batch_size, output_layout);
			//printf("[last forward] CM write out %d, batch %d\n", (threadIdx.z + idx * BLOCK_DIM_Z) * 16, batch_size);
		}
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) {
	// output_threadblock will be filled by the thread block's act_shmem

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	__syncthreads();

	#pragma unroll
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&output_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////


template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, bool INFERENCE>
__global__ void kernel_mlp_fused(
    const Activation activation, 
    const Activation output_activation,
    const __half* __restrict__ input,
    const __half* __restrict__ weights,
    OUT_T* __restrict__ out_intermediate,
    OUT_T* __restrict__ out,
    const uint32_t batch_size,
    const uint32_t in_width,
    const uint32_t out_width,
    const uint32_t n_hidden_matmuls,
    const nvcuda::wmma::layout_t output_layout = nvcuda::wmma::mem_row_major
) {
	// `input` points to the input matrix. Can be any width.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
	// `out` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)

	// if (threadIdx.x == 0) printf("[forward] call kernel_mlp_fused\n");
	// if (threadIdx.x == 0) printf("[forward] inputs=%f\n", (float)input[0]);
	// if (threadIdx.x == 0) printf("[forward] weights=%f\n", (float)weights[0]);

	//if (threadIdx.x == 0) printf("[forward] forward_buffer=%f\n", (float)out_intermediate[0]);

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// In some cases, it also contains the weight matrix for the first and last layer.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	// Each block computes exactly one 16-element chunk of the batch.
	const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS * BLOCK_DIM_Z;

	// First layer
	if (in_width == WIDTH) {
		// If the input has the same width as the network, we can simply use the network's regular layer routine (with static size)
		// instead of using the slower dynamic input layer routine.
		threadblock_load_input_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(activation, act_shmem, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
	} else {
		threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(activation, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width);
	}

	// if (threadIdx.x == 0) printf("[forward] kernel_mlp_fused: passed first layer\n");
	//if (threadIdx.x == 0) printf("[forward] forward_buffer=%f\n", (float)out_intermediate[0]);

	const uint32_t first_layer_size = WIDTH * in_width;
	const uint32_t layer_stride = WIDTH * WIDTH;
	const uint32_t output_stride = WIDTH * batch_size;

	// Hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(activation, act_shmem, weights + first_layer_size + layer_stride * k, !INFERENCE ? (out_intermediate + output_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
		// if (threadIdx.x == 0) printf("[forward] kernel_mlp_fused: passed %d layer\n", k + 1);
		//if (threadIdx.x == 0) printf("[forward] forward_buffer=%f\n", (float)out_intermediate[0]);
	}

	if (out_width > 16) {
		// In the forward pass, intermediate activations are already written out.
		if (INFERENCE) {
			threadblock_write_output_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
		}
	} else if (out) {
		// Last layer
		if (output_layout == nvcuda::wmma::mem_row_major) {
			//printf("[last layer] RM write to out %d\n", elem_idx * 16);
			//if (threadIdx.x == 0) printf("[forward] forward_buffer=%f\n", (float)out_intermediate[0]);
			threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out + elem_idx * 16, 16, output_layout);
			//if (threadIdx.x == 0) printf("[forward] forward_buffer=%f\n", (float)out_intermediate[0]);
		} else {
			//printf("[last layer] CM write to out %d\n", elem_idx);
			//if (threadIdx.x == 0) printf("[forward] forward_buffer=%f\n", (float)out_intermediate[0]);
			threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out + elem_idx, batch_size, output_layout);
			//if (threadIdx.x == 0) printf("[forward] forward_buffer=%f\n", (float)out_intermediate[0]);
		}
	}
}


template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUTPUT_LAYOUT>
__global__ void kernel_mlp_fused_backward(
	const Activation activation, 
	const __half* __restrict__ dL_doutput, 
	const __half* __restrict__ weights,
	__half* __restrict__ out_intermediate,
	const __half* __restrict__ forward,
	__half* __restrict__ dL_dinput,
	const __half* __restrict__ weights_first_layer,
	const uint32_t batch_size,
	const uint32_t out_width,
	const uint32_t n_hidden_matmuls
) {
	// `dL_doutput` points to the input matrix of the backward pass, i.e. the loss gradients. Assumed to be 16 neurons wide.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where backpropagated activation gradients should be written.
	// `forward` points to the memory where the intermediate activations of the forward pass are located. (needed for activation backprop)

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")
	const uint32_t bi = blockIdx.x;	 // block index

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// A skew is applied to the matrix storage to avoid bank conflicts.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	// Multipying one 16-row chunk of intermediate activations with the weight matrix requires all warps of the block.
	// Thus, each block computes exactly one 16-row chunk of the next layer's intermediate activations.
	const uint32_t elem_idx_base = 16 * bi * N_ITERS * BLOCK_DIM_Z;
	const uint32_t elem_idx = elem_idx_base + 16 * threadIdx.z;

	const uint32_t layer_stride = WIDTH * WIDTH;
	const uint32_t output_stride = WIDTH * batch_size;

	// Backprop through last layer
	if (out_width <= 16) {
		using namespace nvcuda;

		// Fragments in registers
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, OUTPUT_LAYOUT> act_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> weights_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag[N_ITERS];

		// Load the relevant chunk of the last layer's weight matrix from global memory into registers
		const uint32_t weights_col = 16 * wi;

		wmma::load_matrix_sync(weights_frag, weights + layer_stride * n_hidden_matmuls + weights_col, WIDTH);

		#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			wmma::fill_fragment(result_frag[l], 0.0f);

			// Load a chunk of output gradients from shared memory and multiply with previously loaded weights
			if (std::is_same<OUTPUT_LAYOUT, wmma::row_major>::value) {
				wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * 16, 16);
			} else {
				wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * (threadIdx.z + l * BLOCK_DIM_Z)), batch_size);
			}

			// NOTE: activation transfer of the _output_ activation is expected to be done _prior_ to calling this kernel
			//       in a separate pass, because the tranfered activation gradient is also needed to compute the weight
			//       gradient of the last weight matrix (see backward()).
			wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

			// Load the temporary forward matrix for the relu transfer
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> forward_frag;
			wmma::load_matrix_sync(forward_frag, forward + output_stride * n_hidden_matmuls + weights_col + (elem_idx + l * BLOCK_DIM_Z * 16) * WIDTH, WIDTH);

			warp_activation_backward<__half>(activation, result_frag[l], forward_frag, result_frag[l]);
		}

		__syncthreads();

		#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			wmma::store_matrix_sync(act_shmem + weights_col + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
		}

		__syncthreads();

		#pragma unroll
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate[lane_offset + (row + elem_idx + i * BLOCK_DIM_Z * 16) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
		}
	} else {
		// If the output width is larger than 16, we will have used CUTLASS for backpropping through the last layer.
		// Load the resulting gradients.
		threadblock_load_input_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
	}

	// Backprop through hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, true>(activation, act_shmem, weights + layer_stride * (n_hidden_matmuls - k - 1), out_intermediate + output_stride * (k + 1) + elem_idx_base * WIDTH, forward + output_stride * (n_hidden_matmuls - k - 1) + elem_idx_base * WIDTH);
	}

	// Compute loss gradients w.r.t. input if desired.
	// THIS CODE ASSUMES THAT THE INPUT WIDTH IS THE SAME AS THE NETWORK WIDTH.
	// DON'T PASS A NON-NULL dL_dinput IF THIS REQUIREMENT IS NOT MET.
	if (dL_dinput != nullptr) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, true>(Activation::None, act_shmem, weights_first_layer, dL_dinput + elem_idx_base * WIDTH);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////


template <uint32_t WIDTH, bool INFERENCE> // WIDTH is hidden_dim
void ffmlp_forward_cuda(
	const __half *inputs, 
	const __half *weights, 
	const uint32_t B, 
	const uint32_t input_dim,
	const uint32_t output_dim,
	const uint32_t num_layers,
	const Activation activation,
	const Activation output_activation,
	__half *forward_buffer,
	__half *outputs
) {

    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
	constexpr uint32_t INPUT_SKEW = 8; // <- likewise with inputs
	constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;
    
    const int N_ITERS = WIDTH >= 256 ? 2 : 8;
	const uint32_t BLOCK_DIM_Z = (INFERENCE && WIDTH == 128) ? 2 : 1;

    const dim3 threads = { 32u, N_BLOCK_ROWS, BLOCK_DIM_Z }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

	uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
	uint32_t n_blocks = div_round_up(B, n_elems_per_block);

    size_t shmem_size = sizeof(__half) * (16 + 16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW); // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*BLOCK_DIM_Z*N_ITERS rows of intermediate activations

	// If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
	if (input_dim != WIDTH) {
		shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16 * BLOCK_DIM_Z) * (input_dim + INPUT_SKEW));
	}

	//printf("[ffmlp_forward_cuda] shmem size = %d\n", shmem_size);

    const dim3 blocks = { n_blocks, 1u, 1u };

    check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, INFERENCE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));

	kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, INFERENCE><<<blocks, threads, shmem_size, 0>>>(
        activation,
		output_activation,
		inputs, // CM
		weights, // RM
		forward_buffer, // CM
		outputs, // CM
		B,
		input_dim,
		output_dim,
		num_layers - 1,
		nvcuda::wmma::mem_row_major // reversed outputs's layout
	);
}


template <uint32_t WIDTH> // WIDTH is hidden_dim
void ffmlp_backward_cuda(
	const __half *grad, 
	const __half *weights, 
	const uint32_t B, 
	const uint32_t input_dim, 
	const uint32_t output_dim, 
	const uint32_t num_layers, 
	const Activation activation,
	const __half *forward_buffer, 
	__half *backward_buffer, 
	__half *grad_inputs
) {

	// locate
	const __half * weights_first = weights;
	const __half * weights_second = weights + input_dim * WIDTH;

    constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
	constexpr uint32_t N_BLOCKS = WIDTH / 16;
    
    const int N_ITERS = WIDTH >= 256 ? 2 : 8;
	const uint32_t BLOCK_DIM_Z = 1;

    const dim3 threads = { 32u, N_BLOCKS, BLOCK_DIM_Z }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

	uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
	uint32_t n_blocks = div_round_up(B, n_elems_per_block);

    size_t shmem_size = sizeof(__half) * ((16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW)); // WIDTH rows of input and 16 * threads.z rows of weights

    const dim3 blocks = { n_blocks, 1u, 1u };

	// The kernels operate with transposed layouts compared with the MLP code
    check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, nvcuda::wmma::row_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

	kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, nvcuda::wmma::row_major><<<blocks, threads, shmem_size, 0>>>(
		activation,
		grad, // CM
		weights_second, // RM
		backward_buffer, // CM
		forward_buffer, // CM
		grad_inputs, // CM
		weights_first, // RM
		B, 
		output_dim, 
		num_layers - 1
	);
}


// inputs: col-major [input_dim, B]
// weights: row-major [hidden_dim * input_dim] + [hidden_dim * hidden_dim * (num_layers - 1)] + [output_dim * hidden_dim]
// forward_buffer: col-major [num_layers, hidden_dim, B]
// outputs: col-major [output_dim, B]
void ffmlp_forward(const at::Tensor inputs, const at::Tensor weights, const uint32_t B, const uint32_t input_dim, const uint32_t output_dim, const uint32_t hidden_dim, const uint32_t num_layers, const uint32_t activation_, const uint32_t output_activation_, at::Tensor forward_buffer, at::Tensor outputs) {
    CHECK_CUDA(inputs);
    CHECK_CONTIGUOUS(inputs);
    CHECK_IS_HALF(inputs);

    CHECK_CUDA(weights);
    CHECK_CONTIGUOUS(weights);
    CHECK_IS_HALF(weights);

    Activation activation = convert_activation(activation_);
    Activation output_activation = convert_activation(output_activation_);

	auto inputs_ptr = reinterpret_cast<__half*>(inputs.data_ptr<at::Half>());
	auto weights_ptr = reinterpret_cast<__half*>(weights.data_ptr<at::Half>());
	auto forward_buffer_ptr = reinterpret_cast<__half*>(forward_buffer.data_ptr<at::Half>());
	auto outputs_ptr = reinterpret_cast<__half*>(outputs.data_ptr<at::Half>());

    switch (hidden_dim) {
        case 16: ffmlp_forward_cuda<16, false>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, forward_buffer_ptr, outputs_ptr); break;
        case 32: ffmlp_forward_cuda<32, false>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, forward_buffer_ptr, outputs_ptr); break;
        case 64: ffmlp_forward_cuda<64, false>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, forward_buffer_ptr, outputs_ptr); break;
        case 128: ffmlp_forward_cuda<128, false>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, forward_buffer_ptr, outputs_ptr); break;
        case 256: ffmlp_forward_cuda<256, false>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, forward_buffer_ptr, outputs_ptr); break;
        default: throw std::runtime_error{"hidden_dim should in [16, 32, 64, 128, 256]"};
    }

	// for output_dim > 16 
	if (output_dim > 16) {
		fc_multiply<LastLayer, true, false, false>(0,
			output_dim, hidden_dim, B,
			(weights_ptr + hidden_dim * input_dim + (num_layers - 1) * hidden_dim * hidden_dim), // row-major, [output_dim, hidden_dim]
			(forward_buffer_ptr + (num_layers - 1) * hidden_dim * B), // col-major [hidden_dim, B]
			outputs_ptr, // col-major [outupt_dim, B]
			output_activation
		);
	}
}

void ffmlp_inference(const at::Tensor inputs, const at::Tensor weights, const uint32_t B, const uint32_t input_dim, const uint32_t output_dim, const uint32_t hidden_dim, const uint32_t num_layers, const uint32_t activation_, const uint32_t output_activation_, at::Tensor inference_buffer, at::Tensor outputs) {
    CHECK_CUDA(inputs);
    CHECK_CONTIGUOUS(inputs);
    CHECK_IS_HALF(inputs);

    CHECK_CUDA(weights);
    CHECK_CONTIGUOUS(weights);
    CHECK_IS_HALF(weights);

    Activation activation = convert_activation(activation_);
    Activation output_activation = convert_activation(output_activation_);

	auto inputs_ptr = reinterpret_cast<__half*>(inputs.data_ptr<at::Half>());
	auto weights_ptr = reinterpret_cast<__half*>(weights.data_ptr<at::Half>());
	auto inference_buffer_ptr = reinterpret_cast<__half*>(inference_buffer.data_ptr<at::Half>());
	auto outputs_ptr = reinterpret_cast<__half*>(outputs.data_ptr<at::Half>());

  	switch (hidden_dim) {
        case 16: ffmlp_forward_cuda<16, true>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, inference_buffer_ptr, outputs_ptr); break;
        case 32: ffmlp_forward_cuda<32, true>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, inference_buffer_ptr, outputs_ptr); break;
        case 64: ffmlp_forward_cuda<64, true>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, inference_buffer_ptr, outputs_ptr); break;
        case 128: ffmlp_forward_cuda<128, true>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, inference_buffer_ptr, outputs_ptr); break;
        case 256: ffmlp_forward_cuda<256, true>(inputs_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, output_activation, inference_buffer_ptr, outputs_ptr); break;
        default: throw std::runtime_error{"hidden_dim should in [16, 32, 64, 128, 256]"};
    }

	// for output_dim > 16 
	if (output_dim > 16) {
		fc_multiply<LastLayer, true, false, false>(0,
			output_dim, hidden_dim, B,
			(weights_ptr + hidden_dim * input_dim + (num_layers - 1) * hidden_dim * hidden_dim), // row-major, [output_dim, hidden_dim]
			inference_buffer_ptr, // col-major [hidden_dim, B]
			outputs_ptr, // col-major [outupt_dim, B]
			output_activation
		);
	}
}

inline std::vector<cudaStream_t>& streams_splitk() {
	static std::vector<cudaStream_t> res;
	return res;
}

inline std::vector<cudaEvent_t>& events_splitk() {
	static std::vector<cudaEvent_t> res;
	return res;
}

void allocate_splitk(size_t size) {
	auto& streams = streams_splitk();
	auto& events = events_splitk();
	streams.resize(size);
	events.resize(size);
	for (size_t i = 0; i < size; i++) {
		CUDA_CHECK_THROW(cudaStreamCreate(&streams[i]));
		CUDA_CHECK_THROW(cudaEventCreate(&events[i]));
	}
}

void free_splitk() {
	auto& streams = streams_splitk();
	auto& events = events_splitk();
	for (size_t i = 0; i < streams.size(); i++) {
		cutlass_free_workspace(streams[i]);
		CUDA_CHECK_PRINT(cudaStreamDestroy(streams[i]));
		CUDA_CHECK_PRINT(cudaEventDestroy(events[i]));
	}
}

// grad: col-major [output_dim, B]
// inputs: col-major [input_dim, B]
// weights: row-major [hidden_dim * input_dim] + [hidden_dim * hidden_dim * (num_layers - 1)] + [output_dim * hidden_dim]
// forward_buffer: col-major [num_layers, hidden_dim, B]
// backward_buffer: col-major [num_layers, hidden_dim, B]
// grad_inputs: col-major [input_dim, B]
// grad_weights: row-major [hidden_dim * input_dim] + [hidden_dim * hidden_dim * (num_layers - 1)] + [output_dim * hidden_dim]
void ffmlp_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor weights, const at::Tensor forward_buffer, const uint32_t B, const uint32_t input_dim, const uint32_t output_dim, const uint32_t hidden_dim, const uint32_t num_layers, const uint32_t activation_, const uint32_t output_activation_, const bool calc_grad_inputs, at::Tensor backward_buffer, at::Tensor grad_inputs, at::Tensor grad_weights) {
	CHECK_CUDA(grad);
    CHECK_CONTIGUOUS(grad);
    CHECK_IS_HALF(grad);

	CHECK_CUDA(inputs);
    CHECK_CONTIGUOUS(inputs);
    CHECK_IS_HALF(inputs);

	CHECK_CUDA(weights);
    CHECK_CONTIGUOUS(weights);
    CHECK_IS_HALF(weights);

	CHECK_CUDA(forward_buffer);
    CHECK_CONTIGUOUS(forward_buffer);
    CHECK_IS_HALF(forward_buffer);	

	CHECK_CUDA(backward_buffer);
    CHECK_CONTIGUOUS(backward_buffer);
    CHECK_IS_HALF(backward_buffer);	

	CHECK_CUDA(grad_weights);
    CHECK_CONTIGUOUS(grad_weights);
    CHECK_IS_HALF(grad_weights);

	CHECK_CUDA(grad_inputs);
    CHECK_CONTIGUOUS(grad_inputs);
    CHECK_IS_HALF(grad_inputs);

    Activation activation = convert_activation(activation_);
    Activation output_activation = convert_activation(output_activation_);

	// activation_backward_output_gpu (I gonna discard output_activation ...)

	int split_k_factor = B / std::min((uint32_t)(1 << 12), B);

	uint32_t forward_index = num_layers - 1;
	uint32_t backward_index = 0;

	auto backward_buffer_ptr = reinterpret_cast<__half*>(backward_buffer.data_ptr<at::Half>());
	auto forward_buffer_ptr = reinterpret_cast<__half*>(forward_buffer.data_ptr<at::Half>());
	auto grad_ptr = reinterpret_cast<__half*>(grad.data_ptr<at::Half>());
	auto inputs_ptr = reinterpret_cast<__half*>(inputs.data_ptr<at::Half>());
	auto weights_ptr = reinterpret_cast<__half*>(weights.data_ptr<at::Half>());
	auto grad_weights_ptr = reinterpret_cast<__half*>(grad_weights.data_ptr<at::Half>());
	
	auto grad_inputs_ptr = calc_grad_inputs ? reinterpret_cast<__half*>(grad_inputs.data_ptr<at::Half>()) : nullptr;
	auto grad_inputs_fused_ptr = input_dim == hidden_dim ? grad_inputs_ptr : nullptr; 



	// calc output layer, grad_weights
	cudaEventRecord(events_splitk().at(backward_index), 0);
	cudaStreamWaitEvent(streams_splitk().at(backward_index), events_splitk().at(backward_index), 0);

	fc_multiply_split_k<LastLayerK, false, true, true>(streams_splitk().at(backward_index), 
		output_dim, B, hidden_dim,
		grad_ptr, // col-major, [output_dim, B]
		(forward_buffer_ptr + forward_index * hidden_dim * B), // row-major, [B, hidden_dim]
		(grad_weights_ptr + hidden_dim * input_dim + (num_layers - 1) * hidden_dim * hidden_dim), // row-major, [output_dim, hidden_dim]
		split_k_factor
	);

	cudaEventRecord(events_splitk().at(backward_index), streams_splitk().at(backward_index));

	// prepare the last backward_buffer if output_dim > 16
	if (output_dim > 16) {
		fc_multiply<FullLayer, false, false, false>(0, 
			hidden_dim, output_dim, B,
			(grad_weights_ptr + hidden_dim * input_dim + (num_layers - 1) * hidden_dim * hidden_dim), // col-major, [hidden_dim, output_dim]
			grad_ptr, // col-major, [output_dim, B]
			(forward_buffer_ptr + forward_index * hidden_dim * B), // col-major, [hidden_dim, B]
			(backward_buffer_ptr + backward_index * hidden_dim * B), // col-major [hidden_dim, B]
			activation, 
			true
		);
	}

	// prepare backward_buffer
	// calc grad_inputs if input_dim == hidden_dim
    switch (hidden_dim) {
        case 16: ffmlp_backward_cuda<16>(grad_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, forward_buffer_ptr, backward_buffer_ptr, grad_inputs_fused_ptr); break;
        case 32: ffmlp_backward_cuda<32>(grad_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, forward_buffer_ptr, backward_buffer_ptr, grad_inputs_fused_ptr); break;
        case 64: ffmlp_backward_cuda<64>(grad_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, forward_buffer_ptr, backward_buffer_ptr, grad_inputs_fused_ptr); break;
        case 128: ffmlp_backward_cuda<128>(grad_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, forward_buffer_ptr, backward_buffer_ptr, grad_inputs_fused_ptr); break;
        case 256: ffmlp_backward_cuda<256>(grad_ptr, weights_ptr, B, input_dim, output_dim, num_layers, activation, forward_buffer_ptr, backward_buffer_ptr, grad_inputs_fused_ptr); break;
        default: throw std::runtime_error{"hidden_dim should in [16, 32, 64, 128, 256]"};
    }	

	//printf("[backward] finished backward kernel\n");

	forward_index--;
	backward_index++;

	// calc middle layer's grad_weights
	for (uint32_t i = 0; i < num_layers - 1; i++) {

		uint32_t matrix_index = num_layers - 2 - i;

		cudaEventRecord(events_splitk().at(backward_index), 0);
		cudaStreamWaitEvent(streams_splitk().at(backward_index), events_splitk().at(backward_index), 0);

		fc_multiply_split_k<FullLayerK, false, true, true>(streams_splitk().at(backward_index), 
			hidden_dim, B, hidden_dim,
			(backward_buffer_ptr + (backward_index - 1) * hidden_dim * B), // col-major [hidden_dim, B]
			(forward_buffer_ptr + forward_index * hidden_dim * B), // row-major [B, hidden_dim]
			(grad_weights_ptr + hidden_dim * input_dim + matrix_index * hidden_dim * hidden_dim), // row-major, [hidden_dim, hidden_dim]
			split_k_factor
		);

		cudaEventRecord(events_splitk().at(backward_index), streams_splitk().at(backward_index));
	
		forward_index--;
		backward_index++;
	}

	// calc input layer's grad_weights
	cudaEventRecord(events_splitk().at(backward_index), 0);
	cudaStreamWaitEvent(streams_splitk().at(backward_index), events_splitk().at(backward_index), 0);

	fc_multiply_split_k<FullLayerK, false, true, true>(streams_splitk().at(backward_index), 
		hidden_dim, B, input_dim,
		(backward_buffer_ptr + (backward_index - 1) * hidden_dim * B), // col-major [hidden_dim, B]
		inputs_ptr, // row-major, [B, input_dim]
		grad_weights_ptr, // row-major, [hidden_dim, input_dim]
		split_k_factor
	);

	cudaEventRecord(events_splitk().at(backward_index), streams_splitk().at(backward_index));

	// calc grad_inputs if input_dim != hidden_dim
	if (calc_grad_inputs && grad_inputs_fused_ptr == nullptr) {
		fc_multiply<FullLayer, false, false, false>(0, 
			input_dim, hidden_dim, B,
			weights_ptr, // col-major [input_dim, hidden_dim]
			(backward_buffer_ptr + (backward_index - 1) * hidden_dim * B), // col-major [hidden_dim, B]
			grad_inputs_ptr // col-major [input_dim, B]
		);
	}
	
	// All the per-layer split-k matrix multiplications summing over
	// the batch are computed in parallel streams to the actual
	// backpropagation. Here, we need to wait for all of these to complete.
	for (auto& event : events_splitk()) {
		cudaStreamWaitEvent(0, event, 0);
	}	
}