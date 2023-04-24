#include "nbody.h"

// Helper functions for Barnes-Hut algorithm
OctreeNode *create_octree(System sys);
void insert_body(OctreeNode *node, System sys, long body_index);
void free_octree(OctreeNode *node);
void compute_force(OctreeNode *node, System sys, long target_body_index, data_t delta, double theta);
void acc_barnes_hut(System sys, System buf, data_t delta, double theta);


/* calculate accelerations */
void acc_cpu(System sys) {
    for (long _i = 0; _i < sys.N; _i++) {
        // coordinates of acceleration vector
        data_t ax = 0.0f;
        data_t ay = 0.0f;

        // actual offset in array a, sys.s & sys.v
        long i = _i * NUM_DIMS;

        // calculate acceleration vector
        for (long _j = 0; _j < sys.N; _j++) {
            // skip computing acceleration when _i == _j
            if (_i == _j) continue;

            // actual offset in array a, sys.s & sys.v
            long j = _j * NUM_DIMS;

            // vector that connects object i and j
            data_t rx = sys.s[j] - sys.s[i];
            data_t ry = sys.s[j + 1] - sys.s[i + 1];

            // the euclidean norm of the difference vector
            data_t r_p2 = rx * rx + ry * ry;
            data_t r_p1 = sqrt(r_p2);  

            // raise to the 3rd power, and divides mass
            data_t tmp = sys.m[_j] / (r_p1 * r_p2);

            // update acceleration vector
            ax += tmp * rx;
            ay += tmp * ry;
        }

        sys.a[i] = ax * G_CONST;
        sys.a[i + 1] = ay * G_CONST;
    }
}

/* avx version of acc_cpu */
void acc_avx(System sys) {
    __m256d _g = _mm256_set1_pd(G_CONST);  // [g, g, g, g]
    __m256d _0 = _mm256_set1_pd(0);        // [0, 0, 0, 0]

    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
        // [x0, y0, x0, y0]
        __m256d s00 = _mm256_setr_pd(sys.s[i], sys.s[i + 1], sys.s[i], sys.s[i + 1]);

        // acceleration vectors
        __m256d a12 = _mm256_setzero_pd();
        __m256d a34 = _mm256_setzero_pd();

        // load 4 objects' positions at a time!
        for (long j = 0; j < sys.N * NUM_DIMS; j += NUM_DIMS * 4) {
            // [m1, m3, m2, m4]
            long _j = j / NUM_DIMS;
            __m256d m1324 = _mm256_setr_pd(
                sys.m[_j], sys.m[_j + 2], sys.m[_j + 1], sys.m[_j + 3]
            );

            // [x1, y1, x2, y2] & [x3, y3, x4, y4]
            __m256d s12 = _mm256_loadu_pd(&sys.s[j]);
            __m256d s34 = _mm256_loadu_pd(&sys.s[j + 4]);

            // vector differences
            __m256d d12 = _mm256_sub_pd(s12, s00);
            __m256d d34 = _mm256_sub_pd(s34, s00);

            // euclidean norms
            __m256d d12_p2   = _mm256_mul_pd(d12, d12);
            __m256d d34_p2   = _mm256_mul_pd(d34, d34);
            __m256d n1324_p2 = _mm256_hadd_pd(d12_p2, d34_p2);
            __m256d n1324_p1 = _mm256_sqrt_pd(n1324_p2);

            // raise to the 3rd power
            __m256d n1324_p3 = _mm256_mul_pd(n1324_p1, n1324_p2);

            // mass / norm^3
            __m256d _1324 = _mm256_div_pd(m1324, n1324_p3);

            // replace inf by 0. 
            // similar to if (i == j) continue;
            if (i / NUM_DIMS / 4 == j / NUM_DIMS / 4) {
                long idx = (i / NUM_DIMS) % 4 ;
                if (idx == 0)
                    _1324 = _mm256_blend_pd(_1324, _0, 0b0001);
                else if (idx == 1)
                    _1324 = _mm256_blend_pd(_1324, _0, 0b0100);
                else if (idx == 2)
                    _1324 = _mm256_blend_pd(_1324, _0, 0b0010);
                else
                    _1324 = _mm256_blend_pd(_1324, _0, 0b1000);
            }

            // update acceleration vector
            __m256d _12 = _mm256_permute_pd(_1324, 0b0000);
            __m256d _34 = _mm256_permute_pd(_1324, 0b1111);

            a12 = _mm256_fmadd_pd(_12, d12, a12);
            a34 = _mm256_fmadd_pd(_34, d34, a34);
        }

        // times G_CONST
        a12 = _mm256_mul_pd(a12, _g);
        a34 = _mm256_mul_pd(a34, _g);

        // sum up and push to array
        __m256d sum0 = _mm256_add_pd(a12, a34);
        __m128d sum1 = _mm_add_pd(_mm256_extractf128_pd(sum0, 0), _mm256_extractf128_pd(sum0, 1));

        // write to array
        _mm_storeu_pd(&sys.a[i], sum1);
    }
}

/* multithreading version of acc_avx */
void acc_omp(System sys) {
    __m256d _g = _mm256_set1_pd(G_CONST);  // [g, g, g, g]
    __m256d _0 = _mm256_set1_pd(0);        // [0, 0, 0, 0]

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
        // [x0, y0, x0, y0]
        __m256d s00 = _mm256_setr_pd(sys.s[i], sys.s[i + 1], sys.s[i], sys.s[i + 1]);

        // acceleration vectors
        __m256d a12 = _mm256_setzero_pd();
        __m256d a34 = _mm256_setzero_pd();

        // load 4 objects' positions at a time!
        for (long j = 0; j < sys.N * NUM_DIMS; j += NUM_DIMS * 4) {
            // [m1, m3, m2, m4]
            long _j = j / NUM_DIMS;
            __m256d m1324 = _mm256_setr_pd(
                sys.m[_j], sys.m[_j + 2], sys.m[_j + 1], sys.m[_j + 3]
            );

            // [x1, y1, x2, y2] & [x3, y3, x4, y4]
            __m256d s12 = _mm256_loadu_pd(&sys.s[j]);
            __m256d s34 = _mm256_loadu_pd(&sys.s[j + 4]);

            // vector differences
            __m256d d12 = _mm256_sub_pd(s12, s00);
            __m256d d34 = _mm256_sub_pd(s34, s00);

            // euclidean norms
            __m256d d12_p2   = _mm256_mul_pd(d12, d12);
            __m256d d34_p2   = _mm256_mul_pd(d34, d34);
            __m256d n1324_p2 = _mm256_hadd_pd(d12_p2, d34_p2);
            __m256d n1324_p1 = _mm256_sqrt_pd(n1324_p2);

            // raise to the 3rd power
            __m256d n1324_p3 = _mm256_mul_pd(n1324_p1, n1324_p2);

            // mass / norm^3
            __m256d _1324 = _mm256_div_pd(m1324, n1324_p3);

            // replace inf by 0. 
            // similar to if (i == j) continue;
            if (i / NUM_DIMS / 4 == j / NUM_DIMS / 4) {
                long idx = (i / NUM_DIMS) % 4 ;
                if (idx == 0)
                    _1324 = _mm256_blend_pd(_1324, _0, 0b0001);
                else if (idx == 1)
                    _1324 = _mm256_blend_pd(_1324, _0, 0b0100);
                else if (idx == 2)
                    _1324 = _mm256_blend_pd(_1324, _0, 0b0010);
                else
                    _1324 = _mm256_blend_pd(_1324, _0, 0b1000);
            }

            // update acceleration vector
            __m256d _12 = _mm256_permute_pd(_1324, 0b0000);
            __m256d _34 = _mm256_permute_pd(_1324, 0b1111);

            a12 = _mm256_fmadd_pd(_12, d12, a12);
            a34 = _mm256_fmadd_pd(_34, d34, a34);
        }

        // times G_CONST
        a12 = _mm256_mul_pd(a12, _g);
        a34 = _mm256_mul_pd(a34, _g);

        // sum up and push to array
        __m256d sum0 = _mm256_add_pd(a12, a34);
        __m128d sum1 = _mm_add_pd(_mm256_extractf128_pd(sum0, 0), _mm256_extractf128_pd(sum0, 1));

        // write to array
        _mm_storeu_pd(&sys.a[i], sum1);
    }
}

/* naive implementation */
/* v <- v + d * a */
/* s <- s + d * v */
void sim_cpu_v00(System sys, System buf, data_t delta) {
    // calculate accelerations
    acc_cpu(sys);

    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
        // update positions
        buf.s[i] = sys.s[i] + delta * sys.v[i];
        buf.s[i + 1] = sys.s[i + 1] + delta * sys.v[i + 1];

        // update velocities
        buf.v[i] = sys.v[i] + delta * sys.a[i];
        buf.v[i + 1] = sys.v[i + 1] + delta * sys.a[i + 1];
    }
}

/* avx256 version of sim_cpu_v00 */
void sim_avx_v00(System sys, System buf, data_t delta) {
    // calculate accelerations
    acc_avx(sys);

    __m256d _d = _mm256_set1_pd(delta);
    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS * 2) {
        __m256d s12_old = _mm256_loadu_pd(&sys.s[i]);
        __m256d v12_old = _mm256_loadu_pd(&sys.v[i]);
        __m256d a12     = _mm256_loadu_pd(&sys.a[i]);

        // update positions and velocities
        __m256d s12_new = _mm256_fmadd_pd(v12_old, _d, s12_old);
        __m256d v12_new = _mm256_fmadd_pd(a12    , _d, v12_old);

        // save to buffers
        _mm256_storeu_pd(&buf.s[i], s12_new);
        _mm256_storeu_pd(&buf.v[i], v12_new);
    }
}

/* multithreading version of sim_avx_v00 */
void sim_omp_v00(System sys, System buf, data_t delta) {
    // calculate accelerations
    acc_omp(sys);

    __m256d _d = _mm256_set1_pd(delta);
    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS * 2) {
        __m256d s12_old = _mm256_loadu_pd(&sys.s[i]);
        __m256d v12_old = _mm256_loadu_pd(&sys.v[i]);
        __m256d a12     = _mm256_loadu_pd(&sys.a[i]);

        // update positions and velocities
        __m256d s12_new = _mm256_fmadd_pd(v12_old, _d, s12_old);
        __m256d v12_new = _mm256_fmadd_pd(a12    , _d, v12_old);

        // save to buffers
        _mm256_storeu_pd(&buf.s[i], s12_new);
        _mm256_storeu_pd(&buf.v[i], v12_new);
    }
}

/* based on sim_cpu_v00 */
/* s <- s + d * v + d * d * a / 2 */
void sim_cpu_v01(System sys, System buf, data_t delta) {
    // calculate accelerations
    acc_cpu(sys);

    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
        // update positions
        buf.s[i] = sys.s[i] + delta * sys.v[i] + delta * delta * sys.a[i] * 0.5;
        buf.s[i + 1] = sys.s[i + 1] + delta * sys.v[i + 1] + delta * delta * sys.a[i + 1] * 0.5;

        // update velocities
        buf.v[i] = sys.v[i] + delta * sys.a[i];
        buf.v[i + 1] = sys.v[i + 1] + delta * sys.a[i + 1];
    }
}

/* based on sim_cpu_v01 */
/* leapfrog integration */
void sim_cpu_v02(System sys, System buf, data_t delta, long step) {
    // calculate current accelerations
    if (step == 0)
        acc_cpu(sys);

    // update positions using the previous velocities and current accelerations
    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
        buf.s[i] = sys.s[i] + delta * sys.v[i] + delta * delta * sys.a[i] * 0.5;
        buf.s[i + 1] = sys.s[i + 1] + delta * sys.v[i + 1] + delta * delta * sys.a[i + 1] * 0.5;
    }

    // calculate new accelerations using the updated positions
    acc_cpu(buf);

    // update velocities using the average of the current and new accelerations
    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
        buf.v[i] = sys.v[i] + 0.5 * delta * (sys.a[i] + buf.a[i]);
        buf.v[i + 1] = sys.v[i + 1] + 0.5 * delta * (sys.a[i + 1] + buf.a[i + 1]);
    }
}

/* based on sim_cpu_v01 */
/* Verlet integration */
void sim_cpu_v03(System sys, System buf, data_t delta, long step) {
    // calculate accelerations
    acc_cpu(sys);

    if (step == 0) {
        // initial step needs to involve velocities
        for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
            buf.s[i] = sys.s[i] + delta * sys.v[i] + delta * delta * sys.a[i] * 0.5;
            buf.s[i + 1] = sys.s[i + 1] + delta * sys.v[i + 1] + delta * delta * sys.a[i + 1] * 0.5;
        }
    } else {
        // starting from step 0, velocities are no longer needed
        for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
            buf.s[i] = 2.0 * sys.s[i] - buf.s[i] + delta * delta * sys.a[i];
            buf.s[i + 1] = 2.0 * sys.s[i + 1] - buf.s[i + 1] + delta * delta * sys.a[i + 1];
        }
    }
}

// Helper function to compute the distance squared between two points
data_t distance_squared(data_t *p1, data_t *p2) {
    data_t dist_sq = 0.0;
    for (int d = 0; d < NUM_DIMS; d++) {
        data_t diff = p1[d] - p2[d];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

// Create an OctreeNode
OctreeNode *create_octree_node(data_t lower_bound[NUM_DIMS], data_t upper_bound[NUM_DIMS]) {
    OctreeNode *node = (OctreeNode *)malloc(sizeof(OctreeNode));
    memcpy(node->lower_bound, lower_bound, NUM_DIMS * sizeof(data_t));
    memcpy(node->upper_bound, upper_bound, NUM_DIMS * sizeof(data_t));
    memset(node->children, 0, (1 << NUM_DIMS) * sizeof(OctreeNode *));
    node->mass = 0;
    memset(node->center_of_mass, 0, NUM_DIMS * sizeof(data_t));
    node->is_leaf = 1;
    node->body_index = -1;
    return node;
}

int point_in_node(data_t *point, OctreeNode *node) {
    for (int d = 0; d < NUM_DIMS; d++) {
        if (point[d] < node->lower_bound[d] || point[d] > node->upper_bound[d]) {
            return 0;
        }
    }
    return 1;
}

void split_node(OctreeNode *node, System sys) {
    data_t child_size = (node->upper_bound[0] - node->lower_bound[0]) * 0.5;

    for (int i = 0; i < (1 << NUM_DIMS); i++) {
        data_t lower_bound[NUM_DIMS];
        data_t upper_bound[NUM_DIMS];

        for (int d = 0; d < NUM_DIMS; d++) {
            if (i & (1 << d)) {
                lower_bound[d] = node->lower_bound[d] + child_size;
                upper_bound[d] = node->upper_bound[d];
            } else {
                lower_bound[d] = node->lower_bound[d];
                upper_bound[d] = node->upper_bound[d] - child_size;
            }
        }

        node->children[i] = create_octree_node(lower_bound, upper_bound);
	node->children[i]->mass = node->mass;
	memcpy(node->children[i]->center_of_mass, node->center_of_mass, NUM_DIMS * sizeof(data_t));

    }

    node->is_leaf = 0;
    node->body_index = -1;
}

void update_node_mass_and_com(OctreeNode *node, System sys, long body_index) {
    data_t new_mass = node->mass + sys.m[body_index];
    for (int d = 0; d < NUM_DIMS; d++) {
        node->center_of_mass[d] = (node->mass * node->center_of_mass[d] + sys.m[body_index] * sys.s[body_index * NUM_DIMS + d]) / new_mass;
    }
    node->mass = new_mass;
}

// Insert a body into the Octree
void insert_body(OctreeNode *node, System sys, long body_index) {

    update_node_mass_and_com(node, sys, body_index);
    // If the node is empty, assign the body to the node
    if (node->body_index == -1) {
        node->body_index = body_index;
	node->mass += sys.m[body_index];
	for (int d = 0; d < NUM_DIMS; d++) {
    	    node->center_of_mass[d] += sys.m[body_index] * sys.s[body_index * NUM_DIMS + d];
	}

        return;
    }

    // If the node is not a leaf, find the appropriate child node to insert the body
    if (!node->is_leaf) {
        // Insert the body into the appropriate child node
        for (int i = 0; i < (1 << NUM_DIMS); i++) {
            if (point_in_node(sys.s + body_index * NUM_DIMS, node->children[i])) {
                insert_body(node->children[i], sys, body_index);
                break;
            }
        }
    } else {
        // Split the node into children and insert both bodies
        split_node(node, sys);
        insert_body(node, sys, node->body_index);
        insert_body(node, sys, body_index);
    }
    if (node->mass != 0) {
        for (int d = 0; d < NUM_DIMS; d++) {
            node->center_of_mass[d] /= node->mass;
        }
    }

}


// Free the memory used by the Octree
void free_octree(OctreeNode *node) {
    if (node == NULL) {
        return;
    }

    for (int i = 0; i < (1 << NUM_DIMS); i++) {
        free_octree(node->children[i]);
    }

    free(node);
}

// Compute the force on a body using the Barnes-Hut algorithm
void compute_force(OctreeNode *node, System sys, long target_body_index, data_t delta, double theta) {
    if (node == NULL || (node->is_leaf && node->body_index == target_body_index)) {
        return;
    }

    data_t *target_pos = sys.s + target_body_index * NUM_DIMS;
    data_t *target_acc = sys.a + target_body_index * NUM_DIMS;

    if (node->is_leaf) {
        data_t *source_pos = sys.s + node->body_index * NUM_DIMS;
        data_t dist_sq = distance_squared(target_pos, source_pos);
        data_t dist = sqrt(dist_sq);
	data_t force_mag = G_CONST * sys.m[target_body_index] * sys.m[node->body_index] / dist_sq;

	for (int d = 0; d < NUM_DIMS; d++) {
            target_acc[d] += force_mag * (source_pos[d] - target_pos[d]) / dist;
        }
    } else {
        data_t dist = sqrt(distance_squared(target_pos, node->center_of_mass));
	data_t dist_sq = dist * dist;
        data_t node_size = node->upper_bound[0] - node->lower_bound[0];
        data_t theta_term = node_size / dist;

        if (theta_term <= theta) {
	    data_t force_mag = G_CONST * sys.m[target_body_index] * node->mass / dist_sq;

            for (int d = 0; d < NUM_DIMS; d++) {
		target_acc[d] += force_mag * (node->center_of_mass[d] - target_pos[d]) / dist;
            }
        } else {
            for (int i = 0; i < (1 << NUM_DIMS); i++) {
                compute_force(node->children[i], sys, target_body_index, delta, theta);
            }
        }
    }
}

void update_mass_and_center_of_mass(OctreeNode *node, System sys) {
    if (node == NULL) {
        return;
    }

    if (node->is_leaf) {
        if (node->body_index != -1) {
            node->mass = sys.m[node->body_index];
            memcpy(node->center_of_mass, sys.s + node->body_index * NUM_DIMS, NUM_DIMS * sizeof(data_t));
        }
    } else {
        for (int i = 0; i < (1 << NUM_DIMS); i++) {
            update_mass_and_center_of_mass(node->children[i], sys);
            if (node->children[i]) {
                node->mass += node->children[i]->mass;
                for (int d = 0; d < NUM_DIMS; d++) {
                    node->center_of_mass[d] += node->children[i]->mass * node->children[i]->center_of_mass[d];
                }
            }
        }
        for (int d = 0; d < NUM_DIMS; d++) {
            node->center_of_mass[d] /= node->mass;
        }
    }
}


// Calculate the accelerations using the Barnes-Hut algorithm
void acc_barnes_hut(System sys, System buf, data_t delta, double theta) {
    // Initialize an Octree with the system's bodies
    data_t lower_bound[NUM_DIMS] = {-1, -1};  // Choose appropriate bounds for your system
    data_t upper_bound[NUM_DIMS] = {1, 1};    // Choose appropriate bounds for your system
    OctreeNode *root = create_octree_node(lower_bound, upper_bound);

    for (long i = 0; i < sys.N; i++) {
        insert_body(root, sys, i);
    }

    update_mass_and_center_of_mass(root, sys);

    // Calculate forces using the Barnes-Hut algorithm
    for (long i = 0; i < sys.N; i++) {
        memset(buf.a + i * NUM_DIMS, 0, NUM_DIMS * sizeof(data_t));
        compute_force(root, sys, i, delta, theta);
    }

    // Free the memory used by the Octree
    free_octree(root);
}


/* Barnes-Hut N-Body simulation */
void sim_cpu_v04(System sys, System buf, data_t delta, double theta) {
    // calculate accelerations using the Barnes-Hut algorithm
    acc_barnes_hut(sys, buf, delta, theta);

    // update positions and velocities of the bodies
    for (long i = 0; i < sys.N * NUM_DIMS; i += NUM_DIMS) {
	buf.s[i] = sys.s[i] + delta * buf.v[i] + delta * delta * sys.a[i] * 0.5;
	buf.s[i + 1] = sys.s[i + 1] + delta * buf.v[i + 1] + delta * delta * sys.a[i + 1] * 0.5;
	
	buf.v[i] = sys.v[i] + delta * 0.5 * sys.a[i];
	buf.v[i + 1] = sys.v[i + 1] + delta * 0.5 * sys.a[i + 1];

    }
}

