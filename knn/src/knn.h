#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
// #include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>
#endif



int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{

    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data<float>();
    float *query_dev = query.data<float>();
    long *idx_dev = idx.data<long>();




  if (ref.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    // float *dist_dev = (float*)THCudaMalloc(state, ref_nb * query_nb * sizeof(float));

    // for (int b = 0; b < batch; b++)
    // {
    // // knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
    // //   dist_dev, idx_dev + b * k * query_nb, THCState_getCurrentStream(state));
    //   knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
    //   dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
    // }
    // THCudaFree(state, dist_dev);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("error in knn: %s\n", cudaGetErrorString(err));
    //     THError("aborting");
    // }
    // return 1;

    //============== Replace =====================
    // float *dist_dev = (float*)cudaMalloc(state, ref_nb * query_nb * sizeof(float));
    auto dist_tensor = torch::empty({ref_nb * query_nb}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    float* dist_dev = dist_tensor.data_ptr<float>();

    for (int b = 0; b < batch; b++)
    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(); // 获取当前CUDA流

        knn_device(
            ref_dev + b * dim * ref_nb, 
            ref_nb, query_dev + b * dim * query_nb, 
            query_nb, 
            dim, 
            k, 
            dist_dev, 
            idx_dev + b * k * query_nb, 
            stream // 使用新的获取方法
        );
    }

    // THCudaFree(state, dist_dev); // 不需要手动调用THCudaFree，因为dist_dev是自动管理的torch::Tensor对象。

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in knn: %s\n", cudaGetErrorString(err));
        // THError已被弃用，考虑使用更现代的错误处理方式
        throw std::runtime_error("CUDA error occurred during KNN computation.");
    }
    return 1;

#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }


    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}
