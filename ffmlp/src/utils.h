#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <atomic>
#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")
#define CHECK_IS_HALF(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, #x " must be a Half tensor")

static constexpr uint32_t MIN_GPU_ARCH = 70;

using network_precision_t = __half;

enum class Activation {
	ReLU,
	Exponential,
	Sine,
	Sigmoid,
	Squareplus,
	Softplus,
	None,
};

static constexpr float PI = 3.14159265358979323846f;
static constexpr float SQRT2 = 1.41421356237309504880f;
static constexpr float K_ACT = 10.0f;

__host__ __device__ inline float logistic(const float x) {
	return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ inline float logit(const float x) {
	return -logf(1.0f / (fminf(fmaxf(x, 1e-9f), 1.0f - 1e-9f)) - 1.0f);
}

inline std::atomic<size_t>& total_n_bytes_allocated() {
	static std::atomic<size_t> s_total_n_bytes_allocated{0};
	return s_total_n_bytes_allocated;
}

/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK_THROW(x)                                                                                               \
	do {                                                                                                                  \
		cudaError_t result = x;                                                                                           \
		if (result != cudaSuccess)                                                                                        \
			throw std::runtime_error(std::string("CUDA Error: " #x " failed with error ") + cudaGetErrorString(result));  \
	} while(0)

/// Checks the result of a cudaXXXXXX call and prints an error on failure
#define CUDA_CHECK_PRINT(x)                                                                                   \
	do {                                                                                                      \
		cudaError_t result = x;                                                                               \
		if (result != cudaSuccess)                                                                            \
			std::cout << "CUDA Error: " #x " failed with error " << cudaGetErrorString(result) << std::endl;  \
	} while(0)

#define DEBUG_GUARD_SIZE 0

/// Managed memory on the Device
template<class T>
class GPUMemory {
private:
	T* m_data = nullptr;
	size_t m_size = 0; // Number of elements
	bool m_owned = true;

public:
	GPUMemory() {}

	GPUMemory<T>& operator=(GPUMemory<T>&& other) {
		std::swap(m_data, other.m_data);
		std::swap(m_size, other.m_size);
		return *this;
	}

	GPUMemory(GPUMemory<T>&& other) {
		*this = std::move(other);
	}

	__host__ __device__ GPUMemory(const GPUMemory<T> &other) : m_data{other.m_data}, m_size{other.m_size}, m_owned{false} {}

	void check_guards() const {
#if DEBUG_GUARD_SIZE > 0
		if (!m_data)
			return;
		uint8_t buf[DEBUG_GUARD_SIZE];
		const uint8_t *rawptr=(const uint8_t *)m_data;
		cudaMemcpy(buf, rawptr-DEBUG_GUARD_SIZE, DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
		for (int i=0;i<DEBUG_GUARD_SIZE;++i) if (buf[i] != 0xff) {
			printf("TRASH BEFORE BLOCK offset %d data %p, read 0x%02x expected 0xff!\n", i, m_data, buf[i] );
			break;
		}
		cudaMemcpy(buf, rawptr+m_size*sizeof(T), DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
		for (int i=0;i<DEBUG_GUARD_SIZE;++i) if (buf[i] != 0xfe) {
			printf("TRASH AFTER BLOCK offset %d data %p, read 0x%02x expected 0xfe!\n", i, m_data, buf[i] );
			break;
		}
#endif
	}

	void allocate_memory(size_t n_bytes) {
		if (n_bytes == 0) {
			return;
		}

#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
		std::cout << "GPUMemory: Allocating " << bytes_to_string(n_bytes) << "." << std::endl;
#endif

		uint8_t *rawptr = nullptr;
		CUDA_CHECK_THROW(cudaMalloc(&rawptr, n_bytes+DEBUG_GUARD_SIZE*2));
#if DEBUG_GUARD_SIZE > 0
		CUDA_CHECK_THROW(cudaMemset(rawptr , 0xff, DEBUG_GUARD_SIZE));
		CUDA_CHECK_THROW(cudaMemset(rawptr+n_bytes+DEBUG_GUARD_SIZE , 0xfe, DEBUG_GUARD_SIZE));
#endif
		if (rawptr) rawptr+=DEBUG_GUARD_SIZE;
		m_data=(T*)(rawptr);
		total_n_bytes_allocated() += n_bytes;
	}

	void free_memory() {
		if (!m_data) {
			return;
		}

		uint8_t *rawptr = (uint8_t*)m_data;
		if (rawptr) rawptr-=DEBUG_GUARD_SIZE;
		CUDA_CHECK_THROW(cudaFree(rawptr));

		total_n_bytes_allocated() -= get_bytes();

		m_data = nullptr;
	}

	/// Allocates memory for size items of type T
	GPUMemory(const size_t size) {
		resize(size);
	}

	/// Frees memory again
	__host__ __device__ ~GPUMemory() {
#ifndef __CUDA_ARCH__
		if (!m_owned) {
			return;
		}

		try {
			if (m_data) {
				free_memory();
				m_size = 0;
			}
		} catch (std::runtime_error error) {
			// Don't need to report on memory-free problems when the driver is shutting down.
			if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
				fprintf(stderr, "Could not free memory: %s\n", error.what());
			}
		}
#endif
	}

	/** @name Resizing/enlargement
	 *  @{
	 */
	/// Resizes the array to the exact new size, even if it is already larger
	void resize(const size_t size) {
		if (!m_owned) {
			throw std::runtime_error("Cannot resize non-owned memory.");
		}

		if (m_size != size) {
			if (m_size) {
				try {
					free_memory();
				} catch (std::runtime_error error) {
					throw std::runtime_error(std::string("Could not free memory: ") + error.what());
				}
			}

			if (size > 0) {
				try {
					allocate_memory(size * sizeof(T));
				} catch (std::runtime_error error) {
					throw std::runtime_error(std::string("Could not allocate memory: ") + error.what());
				}
			}

			m_size = size;
		}
	}

	/// Enlarges the array if its size is smaller
	void enlarge(const size_t size) {
		if (size > m_size) {
			resize(size);
		}
	}
	/** @} */

	/** @name Memset
	 *  @{
	 */
	/// Sets the memory of the first num_elements to value
	void memset(const int value, const size_t num_elements, const size_t offset = 0) {
		if (num_elements + offset > m_size) {
			throw std::runtime_error("Could not set memory: Number of elements larger than allocated memory");
		}

		try {
			CUDA_CHECK_THROW(cudaMemset(m_data + offset, value, num_elements * sizeof(T)));
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not set memory: ") + error.what());
		}
	}

	/// Sets the memory of the all elements to value
	void memset(const int value) {
		memset(value, m_size);
	}
	/** @} */

	/** @name Copy operations
	 *  @{
	 */
	/// Copy data of num_elements from the raw pointer on the host
	void copy_from_host(const T* host_data, const size_t num_elements) {
		try {
			CUDA_CHECK_THROW(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from host: ") + error.what());
		}
	}

	/// Copy num_elements from the host vector
	void copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		if (data.size() < num_elements) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_from_host(data.data(), num_elements);
	}

	/// Copies data from the raw host pointer to fill the entire array
	void copy_from_host(const T* data) {
		copy_from_host(data, m_size);
	}

	/// Copies num_elements of data from the raw host pointer after enlarging the array so that everything fits in
	void enlarge_and_copy_from_host(const T* data, const size_t num_elements) {
		enlarge(num_elements);
		copy_from_host(data, num_elements);
	}

	/// Copies num_elements from the host vector after enlarging the array so that everything fits in
	void enlarge_and_copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		enlarge_and_copy_from_host(data.data(), num_elements);
	}

	/// Copies the entire host vector after enlarging the array so that everything fits in
	void enlarge_and_copy_from_host(const std::vector<T>& data) {
		enlarge_and_copy_from_host(data.data(), data.size());
	}

	/// Copies num_elements of data from the raw host pointer after resizing the array
	void resize_and_copy_from_host(const T* data, const size_t num_elements) {
		resize(num_elements);
		copy_from_host(data, num_elements);
	}

	/// Copies num_elements from the host vector after resizing the array
	void resize_and_copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		resize_and_copy_from_host(data.data(), num_elements);
	}

	/// Copies the entire host vector after resizing the array
	void resize_and_copy_from_host(const std::vector<T>& data) {
		resize_and_copy_from_host(data.data(), data.size());
	}

	/// Copies the entire host vector to the device. Fails if there is not enough space available.
	void copy_from_host(const std::vector<T>& data) {
		if (data.size() < m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(m_size) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_from_host(data.data(), m_size);
	}

	/// Copies num_elements of data from the raw host pointer to the device. Fails if there is not enough space available.
	void copy_to_host(T* host_data, const size_t num_elements) const {
		if (num_elements > m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(m_size));
		}
		try {
			CUDA_CHECK_THROW(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy to host: ") + error.what());
		}
	}

	/// Copies num_elements from the device to a vector on the host
	void copy_to_host(std::vector<T>& data, const size_t num_elements) const {
		if (data.size() < num_elements) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(num_elements) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_to_host(data.data(), num_elements);
	}

	/// Copies num_elements from the device to a raw pointer on the host
	void copy_to_host(T* data) const {
		copy_to_host(data, m_size);
	}

	/// Copies all elements from the device to a vector on the host
	void copy_to_host(std::vector<T>& data) const {
		if (data.size() < m_size) {
			throw std::runtime_error(std::string("Trying to copy ") + std::to_string(m_size) + std::string(" elements, but vector size is only ") + std::to_string(data.size()));
		}
		copy_to_host(data.data(), m_size);
	}

	/// Copies data from another device array to this one, automatically resizing it
	void copy_from_device(const GPUMemory<T> &other) {
		if (m_size != other.m_size) {
			resize(other.m_size);
		}

		try {
			CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice));
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from device: ") + error.what());
		}
	}

	/// Copies size elements from another device array to this one, automatically resizing it
	void copy_from_device(const GPUMemory<T> &other, const size_t size) {
		if (m_size < size) {
			resize(size);
		}

		try {
			CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
		} catch (std::runtime_error error) {
			throw std::runtime_error(std::string("Could not copy from device: ") + error.what());
		}
	}

	// Created an (owned) copy of the data
	GPUMemory<T> copy() const {
		GPUMemory<T> result{m_size};
		result.copy_from_device(*this);
		return result;
	}

	T* data() const {
		check_guards();
		return m_data;
	}

	__host__ __device__ T& operator[](size_t idx) const {
#ifdef DEBUG_BUFFER_OVERRUN
		if (idx > m_size) {
			printf("WARNING: buffer overrun of %p at idx %zu\n", idx);
		}
#endif
		return m_data[idx];
	}

	__host__ __device__ T& operator[](uint32_t idx) const {
#ifdef DEBUG_BUFFER_OVERRUN
		if (idx > m_size) {
			printf("WARNING: buffer overrun of %p at idx %u\n", idx);
		}
#endif
		return m_data[idx];
	}

	size_t get_num_elements() const {
		return m_size;
	}

	size_t size() const {
		return get_num_elements();
	}

	size_t get_bytes() const {
		return m_size * sizeof(T);
	}

	size_t bytes() const {
		return get_bytes();
	}
};


inline std::string bytes_to_string(size_t bytes) {
	std::array<std::string, 7> suffixes = {{ "B", "KB", "MB", "GB", "TB", "PB", "EB" }};

	double count = (double)bytes;
	uint32_t i = 0;
	for (; i < suffixes.size() && count >= 1024; ++i) {
		count /= 1024;
	}

	std::ostringstream oss;
	oss.precision(3);
	oss << count << " " << suffixes[i];
	return oss.str();
}


template <typename T, typename fragment_t>
__host__ __device__ void warp_activation(Activation activation, const fragment_t& frag, fragment_t& result) {
	switch (activation) {
		case Activation::ReLU:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)((T)frag.x[t] > (T)0.0f);
			}
			return;
		case Activation::Exponential:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(expf((float)frag.x[t]));
			}
			return;
		case Activation::Sine:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(sinf((float)frag.x[t]));
			}
			return;
		case Activation::Sigmoid:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(logistic((float)frag.x[t]));
			}
			return;
		case Activation::Squareplus:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				float x = (float)frag.x[t] * K_ACT;
				result.x[t] = (T)(0.5f * (x + sqrtf(x * x + 4)) / K_ACT);
			}
			return;
		case Activation::Softplus:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = (T)(logf(expf((float)frag.x[t] * K_ACT) + 1.0f) / K_ACT);
			}
			return;
		case Activation::None: result = frag; return;
		default:
			// Unsupported activation
			// assert(false); // Commented out due to isolated strange side-effects on Windows
			return;
	}
}

template <typename T, typename fragment_t>
__host__ __device__ fragment_t warp_activation(Activation activation, const fragment_t& frag) {
	fragment_t result;
	warp_activation<T>(activation, frag, result);
	return result;
}

template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ void warp_activation_backward_in(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag_in, fragment_t& result) {
	switch (activation) {
		case Activation::ReLU:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(forward_frag_in.x[t] > (T)0.0f);
			}
			return;
		case Activation::Exponential:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(expf(forward_frag_in.x[t]));
			}
			return;
		case Activation::Sine:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(cosf(forward_frag_in.x[t]));
			}
			return;
		case Activation::Sigmoid:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				float x = logistic(forward_frag_in.x[t]);
				result.x[t] = frag.x[t] * (T)(x * (1.0f - x));
			}
			return;
		case Activation::Squareplus:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				float x = (float)forward_frag_in.x[t] * K_ACT;
				float y = 0.5f * (x + sqrtf(x * x + 4));
				result.x[t] = frag.x[t] * (T)(y * y / (y * y + 1));
			}
			return;
		case Activation::Softplus:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				float tmp = expf((float)frag.x[t] * K_ACT);
				result.x[t] = frag.x[t] * (T)(tmp / (tmp + 1));
			}
			return;
		case Activation::None: result = frag; return;
		default:
			// Unsupported activation
			// assert(false); // Commented out due to isolated strange side-effects on Windows
			return;
	}
}

template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ fragment_t warp_activation_backward_in(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag_in) {
	fragment_t result;
	warp_activation_backward_in<T>(activation, frag, forward_frag_in, result);
	return result;
}

template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ void warp_activation_backward(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag, fragment_t& result) {
	switch (activation) {
		case Activation::ReLU:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(forward_frag.x[t] > (T)0.0f);
			}
			return;
		case Activation::Exponential:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * forward_frag.x[t];
			}
			return;
		case Activation::Sine:
			// Sine requires stored pre-activations, which we don't have. We only
			// write out the post-activations.
			// assert(false); // Commented out due to isolated strange side-effects on Windows
			return;
		case Activation::Sigmoid:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(forward_frag.x[t] * ((T)1.0f - forward_frag.x[t]));
			}
			return;
		case Activation::Squareplus:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				float y = (float)forward_frag.x[t] * K_ACT;
				result.x[t] = frag.x[t] * (T)(y * y / (y * y + 1));
			}
			return;
		case Activation::Softplus:
			#pragma unroll
			for (int t=0; t < result.num_elements; t++) {
				result.x[t] = frag.x[t] * (T)(1.0f - expf(-(float)forward_frag.x[t] * K_ACT));
			}
			return;
		case Activation::None: result = frag; return;
		default:
			// Unsupported activation
			// assert(false); // Commented out due to isolated strange side-effects on Windows
			return;
	}
}

template <typename T, typename fragment_t, typename forward_fragment_t>
__host__ __device__ fragment_t warp_activation_backward(Activation activation, const fragment_t& frag, const forward_fragment_t& forward_frag) {
	fragment_t result;
	warp_activation_backward<T>(activation, frag, forward_frag, result);
	return result;
}