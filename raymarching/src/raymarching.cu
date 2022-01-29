#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>

#include "pcg32.h"

inline constexpr __device__ float DENSITY_THRESH() { return 10.0f; }

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(float x) {
	return copysignf(1.0, x);
}

template <typename T>
__host__ __device__ void swap(T& a, T& b) {
	T c(a); a=b; b=c;
}

__global__ void kernel_generate_points(
    const float * __restrict__ rays_o,
    const float * __restrict__ rays_d,  
    const float * __restrict__ grid,
    const float mean_density,
    const int iter_density,
    const float bound,
    const uint32_t N, const uint32_t H, const uint32_t M,
    float * points,
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
    if (near_x > far_x) swap(near_x, far_x);
    float near_y = (-bound - oy) * rdy;
    float far_y = (bound - oy) * rdy;
    if (near_y > far_y) swap(near_y, far_y);
    float near_z = (-bound - oz) * rdz;
    float far_z = (bound - oz) * rdz;
    if (near_z > far_z) swap(near_z, far_z);

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
void generate_points_cuda(const float *rays_o, const float *rays_d, const float *grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, float *points, int *rays, int *counter) {
    static constexpr uint32_t N_THREAD = 256;
    kernel_generate_points<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o, rays_d, grid, mean_density, iter_density, bound, N, H, M, points, rays, counter);
}



__global__ void kernel_accumulate_rays_forward(
    const float * __restrict__ sigmas,
    const float * __restrict__ rgbs,  
    const float * __restrict__ points,  
    const int * __restrict__ rays,
    const float bound,
    const uint32_t M, const uint32_t N,
    float * depth,
    float * image
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
    float T = 1.0f;

    float r = 0, g = 0, b = 0, d = 0;
    float sum_delta = 0; // sum of delta, to calculate the relative depth map.

    while (step < num_steps) {
        // minimal remained transmittence
        if (T < 1e-4f) break;

        const float sigma = sigmas[0];
        const float delta = points[6];
        const float rr = rgbs[0], gg = rgbs[1], bb = rgbs[2];
        const float alpha = 1.0f - __expf(- sigma * delta);
        const float weight = alpha * T;

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
void accumulate_rays_forward_cuda(const float *sigmas, const float *rgbs, const float *points, const int *rays, const float bound, const uint32_t M, const uint32_t N, float *depth, float *image) {
    static constexpr uint32_t N_THREAD = 256;
    kernel_accumulate_rays_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas, rgbs, points, rays, bound, M, N, depth, image);
}



__global__ void kernel_accumulate_rays_backward(
    const float * __restrict__ grad,
    const float * __restrict__ sigmas,
    const float * __restrict__ rgbs,  
    const float * __restrict__ points,  
    const int * __restrict__ rays,
    const float * __restrict__ image,  
    const float bound,
    const uint32_t M, const uint32_t N,
    float * grad_sigmas,
    float * grad_rgbs
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
    float T = 1.0f;

    const float r_final = image[0], g_final = image[1], b_final = image[2];
    float r = 0, g = 0, b = 0;

    while (step < num_steps) {
        
        if (T < 1e-4f) break;

        const float sigma = sigmas[0];
        const float delta = points[6];
        const float rr = rgbs[0], gg = rgbs[1], bb = rgbs[2];
        const float alpha = 1.0f - __expf(- sigma * delta);
        const float weight = alpha * T;

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
void accumulate_rays_backward_cuda(const float *grad, const float *sigmas, const float *rgbs, const float *points, const int *rays, const float *image, const float bound, const uint32_t M, const uint32_t N, float *grad_sigmas, float *grad_rgbs) {
    static constexpr uint32_t N_THREAD = 256;
    kernel_accumulate_rays_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad, sigmas, rgbs, points, rays, image, bound, M, N, grad_sigmas, grad_rgbs);
}