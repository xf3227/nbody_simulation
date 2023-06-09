#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef NBODY_H
#define NBODY_H

#define NUM_THREADS 16 // for OpenMP

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_DIMS 2          // number of spacial dimensions
#define G_CONST 6.6743E-11  // gravitational constant

typedef double data_t;

typedef struct System {
    long N;       // number of bodies
    data_t *m;  // masses
    data_t *s;  // positions
    data_t *v;  // velocities
    data_t *a;  // accelerations
} System;

enum MemLocation {
    HOST,
    DEVICE
};

/* system attributes */
// void sys_energy(data_t *m, data_t *s, data_t *v, data_t *ans);
// void sys_momentum(data_t *m, data_t *s, data_t *v, data_t *ans);

/* different system initial states  */
System sys_empty(long N, enum MemLocation loc);
System sys_copy(System src, enum cudaMemcpyKind kind);
System sys_rand(
    long N,
    data_t m_min, data_t m_max,
    data_t s_min, data_t s_max,
    data_t v_min, data_t v_max
);
System sys_waltz(data_t m0, data_t m1, data_t r);

/* simulations */
void sim_cpu_v00(System sys, System buf, data_t delta);
void sim_avx_v00(System sys, System buf, data_t delta);
void sim_omp_v00(System sys, System buf, data_t delta);
void sim_cpu_v01(System sys, System buf, data_t delta);
void sim_cpu_v02(System sys, System buf, data_t delta, long step);
void sim_cpu_v03(System sys, System buf, data_t delta, long step);

#ifdef __cplusplus
extern "C" {
#endif

void sim_gpu_v00(System sys, System buf, data_t delta);

#ifdef __cplusplus
}
#endif

/* misc */
void swap_ptr(data_t **a, data_t **b);
double wakeup_delay();
void print_m256d(const char *str, __m256d v);

#endif