#include "nbody.h"

#define BLOCK_SIZE 16

/* the kernel of acc_gpu */
__global__ void acc_gpu(const long N, data_t *m, data_t *s, data_t *a) {
    // object index in block
    long ti = threadIdx.x / BLOCK_SIZE;
    long tj = threadIdx.x % BLOCK_SIZE;

    // object index in system
    long _i = blockIdx.x * BLOCK_SIZE + ti;
    long _j = blockIdx.y * BLOCK_SIZE + tj;

    // conditions for skipping
    if (_i == _j || _i >= N || _j >= N) return;

    // offsets in memory
    long ix = _i * NUM_DIMS;
    long iy = _i * NUM_DIMS + 1;
    long jx = _j * NUM_DIMS;
    long jy = _j * NUM_DIMS + 1;

    // vector that connects object i and j
    data_t rx = s[jx] - s[ix];
    data_t ry = s[jy] - s[iy];

    // the euclidean norm of the difference vector
    data_t r_p2 = rx * rx + ry * ry;
    data_t r_p1 = sqrt(r_p2);  

    // raise to the 3rd power, and divides mass
    data_t tmp = m[_j] / (r_p1 * r_p2);

    // update acceleration vector
    atomicAdd(&a[ix], tmp * rx);
    atomicAdd(&a[iy], tmp * ry);
}

/* the kernel of acc_gpu */
__global__ void acc_gpu_shm(const long N, data_t *m, data_t *s, data_t *a) {
    // object index in block
    long ti = threadIdx.x / BLOCK_SIZE;
    long tj = threadIdx.x % BLOCK_SIZE;

    // object index in system
    long _i = blockIdx.x * BLOCK_SIZE + ti;
    long _j = blockIdx.y * BLOCK_SIZE + tj;
    // if (_i == _j || _i >= N || _j >= N) return;

    // declare shared memory
    __shared__ data_t mj_shared[BLOCK_SIZE];
    __shared__ data_t si_shared[BLOCK_SIZE * NUM_DIMS];
    __shared__ data_t sj_shared[BLOCK_SIZE * NUM_DIMS];

    // offsets in shared memory
    long tix = ti * NUM_DIMS;
    long tiy = ti * NUM_DIMS + 1;
    long tjx = tj * NUM_DIMS;
    long tjy = tj * NUM_DIMS + 1;

    if (ti == 0) {
        mj_shared[tj]  = m[_j];
        si_shared[tjx] = s[(_i + tj) * NUM_DIMS];
        si_shared[tjy] = s[(_i + tj) * NUM_DIMS + 1];
        sj_shared[tjx] = s[_j * NUM_DIMS];
        sj_shared[tjy] = s[_j * NUM_DIMS + 1];
    }

    // load shared memory
    // if (ti == 0) {
    //     mj_shared[tj]  = m[_j];
    // } else if (ti == 2) {
    //     si_shared[tjx] = s[(blockIdx.x * BLOCK_SIZE + tj) * NUM_DIMS];
    // } else if (ti == 4) {
    //     si_shared[tjy] = s[(blockIdx.x * BLOCK_SIZE + tj) * NUM_DIMS + 1];
    // } else if (ti == 6) {
    //     sj_shared[tjx] = s[_j * NUM_DIMS];
    // } else if (ti == 8) {
    //     sj_shared[tjy] = s[_j * NUM_DIMS + 1];
    // }
    __syncthreads();

    if (_i == _j || _i >= N || _j >= N) return;

    // vector that connects object i and j
    data_t rx = sj_shared[tjx] - si_shared[tix];
    data_t ry = sj_shared[tjy] - si_shared[tiy];

    // the euclidean norm of the difference vector
    data_t r_p2 = rx * rx + ry * ry;
    data_t r_p1 = sqrt(r_p2);  

    // raise to the 3rd power, and divides mass
    data_t tmp = mj_shared[tj] / (r_p1 * r_p2);

    // update acceleration vector
    // atomicAdd(&a[_i * NUM_DIMS],     tmp * rx);
    // atomicAdd(&a[_i * NUM_DIMS + 1], tmp * ry);
    // a[_i * NUM_DIMS] += tmp * rx;
    // a[_i * NUM_DIMS + 1] += tmp * r;
}

__global__ void vel_gpu(const long N, data_t *sys_v, data_t *sys_a, data_t delta, data_t *buf_v) {
    long _i = blockIdx.x * blockDim.x + threadIdx.x;
    if (_i >= N) return;

    // actual offsets
    long ix = _i * NUM_DIMS;
    long iy = ix + 1;

    buf_v[ix] = sys_v[ix] + delta * sys_a[ix] * G_CONST;
    buf_v[iy] = sys_v[iy] + delta * sys_a[iy] * G_CONST;
}

__global__ void pos_gpu(const long N, data_t *sys_s, data_t *sys_v, data_t delta, data_t *buf_s) {
    long _i = blockIdx.x * blockDim.x + threadIdx.x;
    if (_i >= N) return;

    // actual offsets
    long ix = _i * NUM_DIMS;
    long iy = ix + 1;

    buf_s[ix] = sys_s[ix] + delta * sys_v[ix];
    buf_s[iy] = sys_s[iy] + delta * sys_v[iy];
}

/* based on sim_cpu_v00 */
extern "C" __host__ void sim_gpu_v00(System sys, System buf, data_t delta) {
    // int numBlocks;        // Occupancy in terms of active blocks
    // int blockSize = 16 * 16;

    // // These variables are used to convert occupancy to warps
    // int device;
    // cudaDeviceProp prop;
    // int activeWarps;
    // int maxWarps;

    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);
    
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks,
    //     vel_gpu,
    //     blockSize,
    //     0);

    // activeWarps = numBlocks * blockSize / prop.warpSize;
    // maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    // printf("Occupancy: %lf%%\n", (double)activeWarps / maxWarps * 100);


    // reset the cuda memory for accelerations
    cudaMemset(sys.a, 0.0, sys.N * NUM_DIMS * sizeof(data_t));

    // calculate accelerations
    dim3 blockDim0(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim0((sys.N - 1) / BLOCK_SIZE + 1, (sys.N - 1) / BLOCK_SIZE + 1);
    acc_gpu_shm<<<gridDim0, blockDim0>>>(sys.N, sys.m, sys.s, sys.a);

    // calcuate positions and velocities
    dim3 blockDim1(512);
    dim3 gridDim1((sys.N - 1) / 512 + 1);
    vel_gpu<<<gridDim1, blockDim1>>>(sys.N, sys.v, sys.a, delta, buf.v);
    pos_gpu<<<gridDim1, blockDim1>>>(sys.N, sys.s, sys.v, delta, buf.s);
}


/* cuBLAS */ 
// extern "C" __host__ void sim_bla_v00(System sys, System buf, data_t delta) {
// }