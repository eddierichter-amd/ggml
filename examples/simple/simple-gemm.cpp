#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// This is a simple model with two tensors a and b
struct simple_model {

    // Parallel GEMM
    struct ggml_tensor * a0;
    struct ggml_tensor * b0;

    // Parallel GEMM
    struct ggml_tensor * a1;
    struct ggml_tensor * b1;

    // the backends used in this compute
    ggml_backend_t backend = NULL;

    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float * a0, float * b0, float *a1, float*b1, int rows_A, int cols_A, int rows_B, int cols_B, bool use_gpu) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    // Create GPU backend
    if (use_gpu) {
      fprintf(stderr, "%s: using CUDA backend\n", __func__);
      model.backend = ggml_backend_cuda_init(0); // init device 0
      if (!model.backend) {
          fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
      }
    }
    else {
      model.backend = ggml_backend_cpu_init();
    }

    // We are doing two GEMMs
    int num_tensors = 4;

    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a0 = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b0 = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);
    model.a1 = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b1 = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a0, a0, 0, ggml_nbytes(model.a0));
    ggml_backend_tensor_set(model.b0, b0, 0, ggml_nbytes(model.b0));
    ggml_backend_tensor_set(model.a1, a1, 0, ggml_nbytes(model.a1));
    ggml_backend_tensor_set(model.b1, b1, 0, ggml_nbytes(model.b1));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    // result = a*b^T
    struct ggml_tensor * result0 = ggml_mul_mat(ctx0, model.a0, model.b0);
    struct ggml_tensor * result1 = ggml_mul_mat(ctx0, model.a1, model.b1);

    // build operations nodes
    ggml_build_forward_expand(gf, result0);
    ggml_build_forward_expand(gf, result1);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
void compute(const simple_model & model, ggml_gallocr_t allocr, struct ggml_tensor **c0, struct ggml_tensor **c1) {
    // reset the allocator to free all the memory allocated during the previous inference

    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    // Returning the last tensors of the graph
    *c0 = ggml_graph_node(gf, -1);
    *c1 = ggml_graph_node(gf, -2);

}

int main(void) {
    ggml_time_init();

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;
    const int rows_B = 3, cols_B = 2;

    float matrix_A0[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    float matrix_B0[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };


    float matrix_A1[rows_A * cols_A] = {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };

    float matrix_B1[rows_B * cols_B] = {
        1, 2,
        3, 4,
        5, 6
    };

    simple_model model;
    load_model(model, matrix_A0, matrix_B0, matrix_A1, matrix_B1, rows_A, cols_A, rows_B, cols_B, true);

    // calculate the temporaly memory required to compute
    ggml_gallocr_t allocr = NULL;

    {
        // Can I just have this use the GPU memory?
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }

    // perform computation
    struct ggml_tensor *c0;
    struct ggml_tensor *c1;
    compute(model, allocr, &c0, &c1);

    // create a array to print result
    std::vector<float> out_data0(ggml_nelements(c0));
    std::vector<float> out_data1(ggml_nelements(c1));

    // bring the data from the backend memory
    ggml_backend_tensor_get(c0, out_data0.data(), 0, ggml_nbytes(c0));
    ggml_backend_tensor_get(c1, out_data1.data(), 0, ggml_nbytes(c1));

    // expected result:
    // [ 60.00 55.00 50.00 110.00
    //  90.00 54.00 54.00 126.00
    //  42.00 29.00 28.00 64.00 ]

    printf("mul mat (%d x %d) (transposed c0):\n[", (int) c0->ne[0], (int) c0->ne[1]);
    for (int j = 0; j < c0->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < c0->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data0[j * c0->ne[0] + i]);
        }
    }
    printf(" ]\n");

    // expected result:
    // [ 5.00 11.00 17.00 23.00
    //  11.00 25.00 39.00 53.00
    //  17.00 39.00 61.00 83.00 ]
    printf("mul mat (%d x %d) (transposed c1):\n[", (int) c1->ne[0], (int) c1->ne[1]);
    for (int j = 0; j < c1->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < c1->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data1[j * c1->ne[0] + i]);
        }
    }
    printf(" ]\n");

    // release backend memory used for computation
    ggml_gallocr_free(allocr);

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
