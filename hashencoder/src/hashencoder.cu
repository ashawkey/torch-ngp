#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t pos_grid[D]) {
	static_assert(D <= 7, "fast_hash can only hash up to 7 dimensions.");

	// While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
	// and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
	// coordinates.
	constexpr uint32_t primes[7] = { 1, 19349663, 83492791, 25165843, 6291469, 12582917, 3145739 };

	uint32_t result = 0;
	#pragma unroll
	for (uint32_t i = 0; i < D; ++i) {
		result ^= pos_grid[i] * primes[i];
	}

	return result;
}


template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index(const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[D]) {
	uint32_t stride = 1;
	uint32_t index = 0;

	#pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        //printf("get_grid_index d=%d, pos_grid[d]=%d, stride=%d, reso=%d\n", d, pos_grid[d], stride, resolution);
        index += pos_grid[d] * stride;
        stride *= (resolution + 1);
    }

    if (stride > hashmap_size) {
        //printf("hash because %d > %d\n", stride, hashmap_size);
        index = fast_hash<D>(pos_grid);
        //printf("hashed (%d, %d) = %d to %d in %d\n", pos_grid[0], pos_grid[1], pos_grid[0] + resolution * pos_grid[1], index % hashmap_size, hashmap_size);
    }

	return (index % hashmap_size) * C + ch;
}


template <uint32_t D, uint32_t C>
__global__ void kernel_grid(
    const float * __restrict__ inputs, 
    const float * __restrict__ grid, 
    const int * __restrict__ offsets, 
    float * outputs, 
    uint32_t B, uint32_t L, uint32_t H,
    const bool calc_grad_inputs, 
    float * dy_dx
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    grid += (uint32_t)offsets[level] * C;
    inputs += b * D;
    outputs += b * L * C + level * C;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    
    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + 0.5f;
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);

        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] += w * grid[index + ch];
        }

        //printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }    

    // prepare dy_dx for calc_grad_inputs
    if (calc_grad_inputs) {

        dy_dx += b * D * L * C + level * D * C; // B L D C

        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {

            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                float w = scale;
                uint32_t pos_grid_local[D];

                #pragma unroll
                for (uint32_t nd = 0; nd < D - 1; nd++) {
                    const uint32_t d = nd > gd ? nd + 1 : nd;

                    if ((idx & (1 << nd)) == 0) {
                        w *= 1 - pos[d];
                        pos_grid_local[d] = pos_grid[d];
                    } else {
                        w *= pos[d];
                        pos_grid_local[d] = pos_grid[d] + 1;
                    }
                }

                pos_grid_local[gd] = pos_grid[gd];
                uint32_t index_left = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
                pos_grid_local[gd] = pos_grid[gd] + 1;
                uint32_t index_right = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);

                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[gd * C + ch] += w * (grid[index_right + ch] - grid[index_left + ch]);
                }
            }
        }
    }
}


template <uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward(
    const float * __restrict__ grad,
    const float * __restrict__ inputs, 
    const float * __restrict__ grid, 
    const int * __restrict__ offsets, 
    float * grad_grid, 
    uint32_t B, uint32_t L, uint32_t H
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
	if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += b * L * C + level * C + ch;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;

    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + 0.5f;
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(ch, hashmap_size, resolution, pos_grid_local);

        #pragma unroll
        for (uint32_t c = 0; c < N_C; c++) {
            atomicAdd(&grad_grid[index + c], w * grad[c]);
        }
    }    
}


template <uint32_t D, uint32_t C>
__global__ void kernel_input_backward(
    const float * __restrict__ grad,
    const float * __restrict__ dy_dx,  
    float * grad_inputs, 
    uint32_t B, uint32_t L
) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= B * D) return;

    const uint32_t b = t / D;
    const uint32_t d = t - b * D;

    grad += b * L * C;
    dy_dx += b * L * D * C;
    
    # pragma unroll
    for (int l = 0; l < L; l++) {
        # pragma unroll
        for (int ch = 0; ch < C; ch++) {
            grad_inputs[t] += grad[l * C + ch] * dy_dx[l * D * C + d * C + ch];
        }
    }
}


template <uint32_t D>
void kernel_grid_wrapper(const float *inputs, const float *embeddings, const int *offsets, float *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, float *dy_dx) {
    static constexpr uint32_t N_THREAD = 512;
	const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid<D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, H, calc_grad_inputs, dy_dx); break;
        case 2: kernel_grid<D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, H, calc_grad_inputs, dy_dx); break;
        case 4: kernel_grid<D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, H, calc_grad_inputs, dy_dx); break;
        case 8: kernel_grid<D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, H, calc_grad_inputs, dy_dx); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [B, L * C], float
// H: base resolution
void encode_forward_cuda(const float *inputs, const float *embeddings, const int *offsets, float *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, float *dy_dx) {
    switch (D) {
        case 2: kernel_grid_wrapper<2>(inputs, embeddings, offsets, outputs, B, C, L, H, calc_grad_inputs, dy_dx); break;
        case 3: kernel_grid_wrapper<3>(inputs, embeddings, offsets, outputs, B, C, L, H, calc_grad_inputs, dy_dx); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
    
}

template <uint32_t D>
void kernel_grid_backward_wrapper(const float *grad, const float *inputs, const float *embeddings, const int *offsets, float *grad_embeddings, const uint32_t B, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, float *dy_dx, float *grad_inputs) {
    static constexpr uint32_t N_THREAD = 256;
	const uint32_t N_C = std::min(2u, C); // n_features_per_thread
	const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    switch (C) {
        case 1: 
            kernel_grid_backward<D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, H); 
            if (calc_grad_inputs) kernel_input_backward<D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 2: 
            kernel_grid_backward<D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, H);
            if (calc_grad_inputs) kernel_input_backward<D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 4: 
            kernel_grid_backward<D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, H);
            if (calc_grad_inputs) kernel_input_backward<D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 8: 
            kernel_grid_backward<D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, H);
            if (calc_grad_inputs) kernel_input_backward<D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


// grad: [B, L * C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// grad_embeddings: [sO, C]
// H: base resolution
void encode_backward_cuda(const float *grad, const float *inputs, const float *embeddings, const int *offsets, float *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, float *dy_dx, float *grad_inputs) {
    switch (D) {
        case 2: kernel_grid_backward_wrapper<2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, H, calc_grad_inputs, dy_dx, grad_inputs); break;
        case 3: kernel_grid_backward_wrapper<3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, H, calc_grad_inputs, dy_dx, grad_inputs); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}
