#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>

#include "pcg32.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float DENSITY_THRESH() { return 10.0f; }


template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


inline __host__ __device__ float signf(float x) {
	return copysignf(1.0, x);
}


template <typename T>
__host__ __device__ void swap_value(T& a, T& b) {
	T c(a); a=b; b=c;
}

template <typename scalar_t>
__global__ void kernel_generate_points(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const scalar_t * __restrict__ grid,
    const float mean_density,
    const int iter_density,
    const float bound,
    const uint32_t N, const uint32_t H, const uint32_t M,
    scalar_t * points,
    int * rays,
    int * counter
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;
    
    const uint32_t max_steps = M / N; // fixed to 1024
    const float rbound = 1 / bound;
    pcg32 rng((uint64_t)n);

    const float density_thresh = fminf(DENSITY_THRESH(), mean_density);

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    // ray marching (naive, no mip, just one level)
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near_x = (-bound - ox) * rdx;
    float far_x = (bound - ox) * rdx;
    if (near_x > far_x) swap_value<float>(near_x, far_x);
    float near_y = (-bound - oy) * rdy;
    float far_y = (bound - oy) * rdy;
    if (near_y > far_y) swap_value<float>(near_y, far_y);
    float near_z = (-bound - oz) * rdz;
    float far_z = (bound - oz) * rdz;
    if (near_z > far_z) swap_value<float>(near_z, far_z);

    const float near = fmaxf(fmaxf(near_x, fmaxf(near_y, near_z)), 0.05f); // hardcoded minimal near distance
    const float far = fminf(far_x, fminf(far_y, far_z));

    const float dt_small = (far - near) / max_steps; // min step size
    const float dt_large = (far - near) / (H - 1); // max step size
    const float cone_angle = 0.5 / (H - 1);

    const float t0 = near + dt_small * rng.next_float();

    // if iter_density too low (thus grid is unreliable), only generate coarse points.
    //if (iter_density < 50) {
    if (false) {

        uint32_t num_steps = H - 1;

        uint32_t point_index = atomicAdd(counter, num_steps);
        uint32_t ray_index = atomicAdd(counter + 1, 1);

        if (point_index + num_steps > M) return;

        points += point_index * 7;

        // write rays
        rays[ray_index * 3] = n;
        rays[ray_index * 3 + 1] = point_index;
        rays[ray_index * 3 + 2] = num_steps;

        float t = t0;
        float last_t = t;
        uint32_t step = 0;

        while (t <= far && step < num_steps) {
            // current point
            const float x = ox + t * dx;
            const float y = oy + t * dy;
            const float z = oz + t * dz;            
            // write step
            points[0] = x;
            points[1] = y;
            points[2] = z;
            points[3] = dx;
            points[4] = dy;
            points[5] = dz;
            step++;
            t += dt_large * rng.next_float() * 2; // random pertube
            points[6] = t - last_t; 
            points += 7;
            last_t = t;
        }
        return;
    }

    // else use two passes to generate fine samples
    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    while (t <= far && num_steps < max_steps) {
        // current point
        const float x = ox + t * dx;
        const float y = oy + t * dy;
        const float z = oz + t * dz;
        // convert to nearest grid position
        const int nx = floorf(0.5 * (x * rbound + 1) * (H - 1)); // (x + bound) / (2 * bound) * (H - 1);
        const int ny = floorf(0.5 * (y * rbound + 1) * (H - 1));
        const int nz = floorf(0.5 * (z * rbound + 1) * (H - 1));

        const uint32_t index = nx * H * H + ny * H + nz;
        const float density = grid[index];

        // if occpuied, advance a small step, and write to output
        if (density > density_thresh) {
            num_steps++;
            const float dt = dt_small; // fmaxf(fminf(t * cone_angle, dt_large), dt_small); 
            //printf("[n=%d s=%d t=%f + %f] occ, p=(%f,%f,%f) n=(%d, %d, %d), density=%f\n", n, num_steps, t, dt, x, y, z, nx, ny, nz, density);
            t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 1 + signf(dx)) / (H - 1) * 2 - 1) * bound - x) * rdx;
            const float ty = (((ny + 1 + signf(dy)) / (H - 1) * 2 - 1) * bound - y) * rdy;
            const float tz = (((nz + 1 + signf(dz)) / (H - 1) * 2 - 1) * bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            //printf("[n=%d s=%d t=%f-->%f] empty, p=(%f,%f,%f) n=(%d, %d, %d), density=%f\n", n, num_steps, t, tt, x, y, z, nx, ny, nz, density);
            do {
                const float dt = dt_small; // fmaxf(fminf(t * cone_angle, dt_large), dt_small); 
                t += dt;
            } while (t < tt);
        }
    }



    //printf("[n=%d] num_steps=%d\n", n, num_steps);
    //printf("[n=%d] num_steps=%d, pc=%d, rc=%d\n", n, num_steps, counter[0], counter[1]);

    // second pass: really locate and write points & dirs
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);

    //printf("[n=%d] num_steps=%d, point_index=%d, ray_index=%d\n", n, num_steps, point_index, ray_index);

    if (num_steps == 0) return;
    if (point_index + num_steps > M) return;

    points += point_index * 7;

    // write rays
    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = num_steps;

    t = t0;
    float last_t = t;
    uint32_t step = 0;

    //rng = pcg32((uint64_t)n); // reset 

    while (t <= far && step < num_steps) {
        // current point
        const float x = ox + t * dx;
        const float y = oy + t * dy;
        const float z = oz + t * dz;
        // convert to nearest grid position
        const int nx = floorf(0.5 * (x * rbound + 1) * (H - 1)); // (x + bound) / (2 * bound) * (H - 1);
        const int ny = floorf(0.5 * (y * rbound + 1) * (H - 1));
        const int nz = floorf(0.5 * (z * rbound + 1) * (H - 1));

        // query grid
        const uint32_t index = nx * H * H + ny * H + nz;
        const float density = grid[index];

        // if occpuied, advance a small step, and write to output
        if (density > density_thresh) {
            // write step
            points[0] = fmaxf(fminf(x, bound), -bound); // clamp again, to make sure in range...
            points[1] = fmaxf(fminf(y, bound), -bound);
            points[2] = fmaxf(fminf(z, bound), -bound);
            points[3] = dx;
            points[4] = dy;
            points[5] = dz;
            step++;
            const float dt = dt_small; // fmaxf(fminf(t * cone_angle, dt_large), dt_small); 
            t += dt;
            points[6] = t - last_t; 
            points += 7;
            last_t = t;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 1 + signf(dx)) / (H - 1) * 2 - 1) * bound - x) * rdx;
            const float ty = (((ny + 1 + signf(dy)) / (H - 1) * 2 - 1) * bound - y) * rdy;
            const float tz = (((nz + 1 + signf(dz)) / (H - 1) * 2 - 1) * bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do {
                const float dt = dt_small; // fmaxf(fminf(t * cone_angle, dt_large), dt_small); 
                t += dt;
            } while (t < tt);
        }
    }
}

// rays_o/d: [N, 3]
// grid: [H, H, H]
// points: [M, 3]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
template <typename scalar_t>
void generate_points_cuda(const scalar_t *rays_o, const scalar_t *rays_d, const scalar_t *grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, scalar_t *points, int *rays, int *counter) {
    static constexpr uint32_t N_THREAD = 256;
    kernel_generate_points<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o, rays_d, grid, mean_density, iter_density, bound, N, H, M, points, rays, counter);
}


template <typename scalar_t>
__global__ void kernel_accumulate_rays_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ points,  
    const int * __restrict__ rays,
    const float bound,
    const uint32_t M, const uint32_t N,
    scalar_t * depth,
    scalar_t * image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    sigmas += offset;
    rgbs += offset * 3;
    points += offset * 7;

    // accumulate 
    uint32_t step = 0;
    scalar_t T = 1.0f;

    scalar_t r = 0, g = 0, b = 0, d = 0;
    scalar_t sum_delta = 0; // sum of delta, to calculate the relative depth map.

    while (step < num_steps) {
        // minimal remained transmittence
        if (T < 1e-4f) break;

        const scalar_t sigma = sigmas[0];
        const scalar_t delta = points[6];
        const scalar_t rr = rgbs[0], gg = rgbs[1], bb = rgbs[2];
        const scalar_t alpha = 1.0f - __expf(- sigma * delta);
        const scalar_t weight = alpha * T;

        d += weight * sum_delta;
        r += weight * rr;
        g += weight * gg;
        b += weight * bb;

        sum_delta += delta;
        T *= 1.0f - alpha;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas++;
        rgbs += 3;
        points += 7;

        step++;
    }

    d /= (1.0f - T) * (2 * bound * 1.73205080757) + 1e-8; // make sure it is strictly in [0, 1)

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // fixed white background
    if (step == num_steps) {
        r += T;
        g += T;
        b += T;
    }

    // write
    depth[index] = d;
    image[index * 3] = r;
    image[index * 3 + 1] = g;
    image[index * 3 + 2] = b;
}

// sigmas: [M]
// rgbs: [M, 3]
// points: [M, 7]
// rays: [N, 3], idx, offset, num_steps
// depth: [N]
// image: [N, 3]
template <typename scalar_t>
void accumulate_rays_forward_cuda(const scalar_t *sigmas, const scalar_t *rgbs, const scalar_t *points, const int *rays, const float bound, const uint32_t M, const uint32_t N, scalar_t *depth, scalar_t *image) {
    static constexpr uint32_t N_THREAD = 256;
    kernel_accumulate_rays_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas, rgbs, points, rays, bound, M, N, depth, image);
}


template <typename scalar_t>
__global__ void kernel_accumulate_rays_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ points,  
    const int * __restrict__ rays,
    const scalar_t * __restrict__ image,  
    const float bound,
    const uint32_t M, const uint32_t N,
    scalar_t * grad_sigmas,
    scalar_t * grad_rgbs
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    grad += index * 3;
    image += index * 3;
    sigmas += offset;
    rgbs += offset * 3;
    points += offset * 7;
    grad_sigmas += offset;
    grad_rgbs += offset * 3;

    //const float loss_scale = 1.0f;

    // accumulate 
    uint32_t step = 0;
    scalar_t T = 1.0f;

    const scalar_t r_final = image[0], g_final = image[1], b_final = image[2];
    scalar_t r = 0, g = 0, b = 0;

    while (step < num_steps) {
        
        if (T < 1e-4f) break;

        const scalar_t sigma = sigmas[0];
        const scalar_t delta = points[6];
        const scalar_t rr = rgbs[0], gg = rgbs[1], bb = rgbs[2];
        const scalar_t alpha = 1.0f - __expf(- sigma * delta);
        const scalar_t weight = alpha * T;

        r += weight * rr;
        g += weight * gg;
        b += weight * bb;

        T *= 1.0f - alpha; // this has been T(t+1)

        // write grad
        grad_rgbs[0] = grad[0] * weight;
        grad_rgbs[1] = grad[1] * weight;
        grad_rgbs[2] = grad[2] * weight;

        grad_sigmas[0] = delta * (
            grad[0] * (T * rr - (r_final - r)) + 
            grad[1] * (T * gg - (g_final - g)) + 
            grad[2] * (T * bb - (b_final - b))
        );
    
        // locate
        sigmas++;
        rgbs += 3;
        grad_sigmas++;
        grad_rgbs += 3;
        points += 7;

        step++;
    }
}


// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// points: [M, 7]
// rays: [N, 3], idx, offset, num_steps
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
void accumulate_rays_backward_cuda(const scalar_t *grad, const scalar_t *sigmas, const scalar_t *rgbs, const scalar_t *points, const int *rays, const scalar_t *image, const float bound, const uint32_t M, const uint32_t N, scalar_t *grad_sigmas, scalar_t *grad_rgbs) {
    static constexpr uint32_t N_THREAD = 256;
    kernel_accumulate_rays_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad, sigmas, rgbs, points, rays, image, bound, M, N, grad_sigmas, grad_rgbs);
}




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

    CHECK_IS_FLOATING(rays_o);
    CHECK_IS_FLOATING(rays_d);
    CHECK_IS_FLOATING(grid);
    CHECK_IS_FLOATING(points);
    CHECK_IS_INT(rays);
    CHECK_IS_INT(counter);


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.type(), "generate_points", ([&] {
        generate_points_cuda<scalar_t>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(), mean_density, iter_density, bound, N, H, M, points.data_ptr<scalar_t>(), rays.data_ptr<int>(), counter.data_ptr<int>());
    }));
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

    CHECK_IS_FLOATING(sigmas);
    CHECK_IS_FLOATING(rgbs);
    CHECK_IS_FLOATING(points);
    CHECK_IS_INT(rays);
    CHECK_IS_FLOATING(depth);
    CHECK_IS_FLOATING(image);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.type(), "accumulate_rays_forward", ([&] {
        accumulate_rays_forward_cuda<scalar_t>(sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), points.data_ptr<scalar_t>(), rays.data_ptr<int>(), bound, M, N, depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
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

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(sigmas);
    CHECK_IS_FLOATING(rgbs);
    CHECK_IS_FLOATING(points);
    CHECK_IS_INT(rays);
    CHECK_IS_FLOATING(image);
    CHECK_IS_FLOATING(grad_sigmas);
    CHECK_IS_FLOATING(grad_rgbs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.type(), "accumulate_rays_backward", ([&] {
        accumulate_rays_backward_cuda<scalar_t>(grad.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), points.data_ptr<scalar_t>(), rays.data_ptr<int>(), image.data_ptr<scalar_t>(), bound, M, N, grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.data_ptr<scalar_t>());
    }));
}