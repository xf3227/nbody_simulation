#include "nbody.h"

#define BLOCK_SIZE 16

/* the kernel of acc_gpu */
__global__ void acc_gpu(const long N, data_t *m, data_t *s, data_t *a) {
    // object index in system
    long _i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    long _j = blockIdx.y * BLOCK_SIZE + threadIdx.y;

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
    // data_t r_p1 = rsqrt(r_p2);
    data_t r_p1 = sqrt(r_p2); 

    // raise to the 3rd power, and divides mass
    // data_t tmp = m[_j] * r_p1 * __frcp_rn(r_p2);
    data_t tmp = m[_j] / (r_p1 * r_p2);

    // update acceleration vector
    atomicAdd(&a[ix], tmp * rx);
    atomicAdd(&a[iy], tmp * ry);
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
    dim3 blockDim0(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim0((sys.N - 1) / BLOCK_SIZE + 1, (sys.N - 1) / BLOCK_SIZE + 1);
    acc_gpu<<<gridDim0, blockDim0>>>(sys.N, sys.m, sys.s, sys.a);

    // calcuate positions and velocities
    dim3 blockDim1(512);
    dim3 gridDim1((sys.N - 1) / 512 + 1);
    vel_gpu<<<gridDim1, blockDim1>>>(sys.N, sys.v, sys.a, delta, buf.v);
    pos_gpu<<<gridDim1, blockDim1>>>(sys.N, sys.s, sys.v, delta, buf.s);
}


/* cuBLAS */ 
// extern "C" __host__ void sim_bla_v00(System sys, System buf, data_t delta) {
// }