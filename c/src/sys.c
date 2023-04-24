#include "nbody.h"

/* generate random number */
data_t fRand(data_t fMin, data_t fMax) {
    data_t f = (data_t)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

/* just allocate memory for a system */
System sys_empty(long N) {
    System sys;
    sys.N = N;
    sys.m = (data_t *) malloc(N * sizeof(data_t));
    sys.s = (data_t *) malloc(N * NUM_DIMS * sizeof(data_t));
    sys.v = (data_t *) malloc(N * NUM_DIMS * sizeof(data_t));
    sys.a = (data_t *) malloc(N * NUM_DIMS * sizeof(data_t));
    return sys;
}

/* generate a copy of the source system */
System sys_copy(System src) {
    System sys = sys_empty(src.N);
    memcpy(sys.m, src.m, src.N * sizeof(data_t));
    memcpy(sys.s, src.s, src.N * NUM_DIMS * sizeof(data_t));
    memcpy(sys.v, src.v, src.N * NUM_DIMS * sizeof(data_t));
    memcpy(sys.a, src.a, src.N * NUM_DIMS * sizeof(data_t)); 
    return sys;
}

/* initialization with random positions and velocities */
System sys_rand(
    long N,
    data_t m_min, data_t m_max,
    data_t s_min, data_t s_max,
    data_t v_min, data_t v_max
) {
    System sys = sys_empty(N);
    
    // initialize with random values
    for (long i = 0; i < N; i++)
        sys.m[i] = fRand(m_min, m_max);
    for (long i = 0; i < N * NUM_DIMS; i++)
        sys.s[i] = fRand(s_min, s_max);
    for (long i = 0; i < N * NUM_DIMS; i++)
        sys.v[i] = fRand(v_min, v_max);

    return sys;
}

/* a 2-body system like the earth and the moon */
/* barycenter is fixed at (0, 0) */
System sys_waltz(data_t m0, data_t m1, data_t r) {
    long N = 2;
    System sys = sys_empty(N);

    // set masses
    sys.m[0] = m0;
    sys.m[1] = m1;

    // distance to barycenter
    data_t r0 = m1 * r / (m0 + m1);
    data_t r1 = m0 * r / (m0 + m1);

    // set positions
    sys.s[0] = r0;
    sys.s[1] = 0.0f;
    sys.s[2] = -r1;
    sys.s[3] = 0.0f;

    // gravitational force between the two objects
    data_t f = G_CONST * m0 * m1 / (r * r);

    // set velocities
    sys.v[0] = 0.0f;
    sys.v[1] = sqrt(f / m0 * r0);
    sys.v[2] = 0.0f;
    sys.v[3] = -sqrt(f / m1 * r1);

    return sys;
}
