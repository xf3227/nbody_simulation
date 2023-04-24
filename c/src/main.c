#include "nbody.h"

#define NUM_BODIES 8192
#define MASS_MIN 1.0e11
#define MASS_MAX 2.0e11
#define POSITION_MIN -1.0
#define POSITION_MAX  1.0
#define VELOCITY_MIN  0.0
#define VELOCITY_MAX  0.0
#define DELTA 1e-4
#define NUM_ITERATIONS 10

int main(int argc, char *argv[]) {
    cudaSetDevice(1);

    double time_start, time_end;
    double wakeup_answer = wakeup_delay();

    // initialize system
    System sys = sys_rand(NUM_BODIES, MASS_MIN, MASS_MAX, POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX);
    System buf = sys_copy(sys, cudaMemcpyHostToHost);


    // printf("m:\t%1.2e %1.2e %1.2e %1.2e\n", sys.m[0], sys.m[1], sys.m[2], sys.m[3]);
    printf("s0x, s0y:\t%1.2e %1.2e\n", sys.s[0], sys.s[1]);
    printf("s1x, s1y:\t%1.2e %1.2e\n", sys.s[2], sys.s[3]);
    // printf("s2x, s2y:\t%1.2e %1.2e\n", sys.s[4], sys.s[5]);
    // printf("s3x, s3y:\t%1.2e %1.2e\n", sys.s[6], sys.s[7]);

    System sys_cuda = sys_copy(sys, cudaMemcpyHostToDevice);
    System buf_cuda = sys_copy(buf, cudaMemcpyHostToDevice);

    // simulation loop
    time_start = omp_get_wtime();
    for (long t = 0; t < NUM_ITERATIONS; t++) {
        sim_avx_v00(sys, buf, DELTA);
        swap_ptr(&sys.s, &buf.s);
        swap_ptr(&sys.v, &buf.v);
        swap_ptr(&sys.a, &buf.a);
    }
    time_end = omp_get_wtime();

    printf("cpu execution time per iteration: %f ms\n", (float)((time_end - time_start) * 1e3 / NUM_ITERATIONS));
    printf("(sys.s[ 0], sys.s[ 1]): (%f, %f)\n", sys.s[0], sys.s[1]);
    printf("(sys.s[-2], sys.s[-1]): (%f, %f)\n", sys.s[sys.N - 2], sys.s[sys.N - 1]);

    cudaDeviceSynchronize();
    time_start = omp_get_wtime();
    for (long t = 0; t < NUM_ITERATIONS; t++) {
        sim_gpu_v00(sys_cuda, buf_cuda, DELTA);
        swap_ptr(&sys_cuda.s, &buf_cuda.s);
        swap_ptr(&sys_cuda.v, &buf_cuda.v);
        swap_ptr(&sys_cuda.a, &buf_cuda.a);
    }
    cudaDeviceSynchronize();
    time_end = omp_get_wtime();

    free(sys.m);
    free(sys.s);
    free(sys.v);
    free(sys.a);
    sys = sys_copy(sys_cuda, cudaMemcpyDeviceToHost);

    printf("gpu execution time per iteration: %f ms\n", (float)((time_end - time_start) * 1e3 / NUM_ITERATIONS));
    printf("(sys.s[ 0], sys.s[ 1]): (%f, %f)\n", sys.s[0], sys.s[1]);
    printf("(sys.s[-2], sys.s[-1]): (%f, %f)\n", sys.s[sys.N - 2], sys.s[sys.N - 1]);
    printf("wakeup delay computed: %g \n", wakeup_answer);
    return 0;
}