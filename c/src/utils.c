#include "nbody.h"

/* calculate total energy in a system */
void sys_energy(data_t *m, data_t *s, data_t *v, data_t *ans) {
    
}

/* calculate net momentum in a system */
void sys_momentum(data_t *m, data_t *s, data_t *v, data_t *ans) {
    
}

/* swap pointers */
void swap_ptr(data_t **a, data_t **b) {
    data_t *tmp = *a;
    *a = *b;
    *b = tmp;
}

/* this routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay() {
    double meas = 0.0f;
    double quasi_random = 0.0f;
    double time_start, time_end;
    long j = 100;
    time_start = omp_get_wtime();
    while (meas < 1.0) {
        for (long i = 1; i < j; i++)
            quasi_random = quasi_random * quasi_random - 1.923432;
        time_end = omp_get_wtime();
        meas = time_end - time_start;
        // twice as much delay next time, until we've taken 1 second
        j *= 2;
    }
    return quasi_random;
}

void print_m256d(const char *str, __m256d v) {
    __m128d vl = _mm256_extractf128_pd(v, 0);
    __m128d vh = _mm256_extractf128_pd(v, 1);
    double arr[4];
    _mm_storeu_pd(&arr[0], vl);
    _mm_storeu_pd(&arr[2], vh);
    printf("%s:\t%1.2e %1.2e %1.2e %1.2e\n", str, arr[0], arr[1], arr[2], arr[3]);
}

// void print_m128d(const char *str, __m128d v) {
//     __m128d vl = _mm256_extractf64_pd(v, 0);
//     __m128d vh = _mm256_extractf64_pd(v, 1);
//     double arr[4];
//     _mm_storeu_pd(&arr[0], vl);
//     _mm_storeu_pd(&arr[2], vh);
//     printf("%s:\t%1.2e %1.2e %1.2e %1.2e\n", str, arr[0], arr[1], arr[2], arr[3]);
// }