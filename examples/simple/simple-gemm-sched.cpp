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

#define N_THREADS 1

// If we want to explicitly place the output on
// the CPU. Otherwise, it will run on the GPU
//#define PLACE_OUTPUT_ON_CPU

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
    std::vector<ggml_backend_t> backends;

    // Scheduler
    ggml_backend_sched_t sched;

    // the backend buffer to storage 
    ggml_backend_buffer_t cpu_buffer;
    ggml_backend_buffer_t gpu_buffer;

    // Allocators
    struct ggml_tallocr cpu_alloc;
    struct ggml_tallocr gpu_alloc;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float * a0, float * b0, float *a1, float*b1, int rows_A, int cols_A, int rows_B, int cols_B) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    ggml_backend_t gpu_backend;
    ggml_backend_t cpu_backend;

    // Create a GPU backend
    gpu_backend = ggml_backend_cuda_init(0); // init device 0
    if (!gpu_backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }

    // Create a CPU backend 
    cpu_backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu_backend, N_THREADS);

    // Pushing back all backends so it can be used by the scheduler
    model.backends.push_back(gpu_backend);
    model.backends.push_back(cpu_backend);
    
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

    // Want to put A0 + B0 on CPU and A1 + B1 on GPU
    model.cpu_buffer = ggml_backend_alloc_buffer(cpu_backend, 8192);
    model.gpu_buffer = ggml_backend_alloc_buffer(gpu_backend, 8192);
    model.cpu_alloc = ggml_tallocr_new(model.cpu_buffer);
    model.gpu_alloc = ggml_tallocr_new(model.gpu_buffer);
    ggml_tallocr_alloc(&model.cpu_alloc, model.a0);
    ggml_tallocr_alloc(&model.cpu_alloc, model.b0);
    ggml_tallocr_alloc(&model.gpu_alloc, model.a1);
    ggml_tallocr_alloc(&model.gpu_alloc, model.b1);
        
    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a0, a0, 0, ggml_nbytes(model.a0));
    ggml_backend_tensor_set(model.b0, b0, 0, ggml_nbytes(model.b0));
    ggml_backend_tensor_set(model.a1, a1, 0, ggml_nbytes(model.a1));
    ggml_backend_tensor_set(model.b1, b1, 0, ggml_nbytes(model.b1));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(simple_model& model) {
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
    struct ggml_tensor * result2 = ggml_mul_mat(ctx0, result0, result1);

    // build operations nodes
    ggml_build_forward_expand(gf, result2);

    // Want to move the compute of result2 to the CPU
#ifdef PLACE_OUTPUT_ON_CPU
    ggml_tallocr_alloc(&model.cpu_alloc, result2);
#endif

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
struct ggml_tensor *compute(simple_model & model, /*ggml_gallocr_t allocr,*/ struct ggml_tensor **c0, struct ggml_tensor **c1) {
    // reset the allocator to free all the memory allocated during the previous inference

    struct ggml_cgraph * gf = build_graph(model);

    ggml_backend_sched_reset(model.sched);
    ggml_backend_sched_graph_compute(model.sched, gf);

    // Returning the last tensors of the graph
    *c0 = ggml_graph_node(gf, -3);
    *c1 = ggml_graph_node(gf, -2);
    
    return ggml_graph_node(gf, -1);

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
    load_model(model, matrix_A0, matrix_B0, matrix_A1, matrix_B1, rows_A, cols_A, rows_B, cols_B);

    model.sched = ggml_backend_sched_new(model.backends.data(), NULL, model.backends.size(), 6 /* graph size */, false);
    struct ggml_cgraph * gf = build_graph(model);

    // Allocating for the scheduler
    // TODO: Does this mean I don't need to allocate from the scheduler??
    ggml_backend_sched_reserve(model.sched, gf);

    // perform computation
    struct ggml_tensor *c0;
    struct ggml_tensor *c1;
    struct ggml_tensor *output = compute(model, /*allocr, */&c0, &c1);

    // Printing the Graph
    ggml_graph_print(gf);

    // create a array to print result
    std::vector<float> out_data0(ggml_nelements(c0));
    std::vector<float> out_data1(ggml_nelements(c1));
    std::vector<float> out_data2(ggml_nelements(output));

    // bring the data from the backend memory
    ggml_backend_tensor_get(c0, out_data0.data(), 0, ggml_nbytes(c0));
    ggml_backend_tensor_get(c1, out_data1.data(), 0, ggml_nbytes(c1));
    ggml_backend_tensor_get(output, out_data2.data(), 0, ggml_nbytes(output));

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

    // expected result:
    //[ 4285.00 4860.00 2477.00
    // 9815.00 11124.00 5671.00
    // 15345.00 17388.00 8865.00 ]
    printf("mul mat (%d x %d) (transposed output):\n[", (int) output->ne[0], (int) output->ne[1]);
    for (int j = 0; j < output->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < output->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data2[j * output->ne[0] + i]);
        }
    }
    printf(" ]\n");


    // release backend memory used for computation
    // TODO: DO I NEED THIS? 
    //ggml_gallocr_free(allocr);

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.gpu_buffer);
    ggml_backend_buffer_free(model.cpu_buffer);
    return 0;
}
