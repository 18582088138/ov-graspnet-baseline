// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "group_points.h"
#include "utils.h"

void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out);

void group_points_kernel_cpu_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out){
    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // 计算当前batch的偏移量
      const float *current_points = points + batch_index * n * c;
      const int *current_idx = idx + batch_index * npoints * nsample;
      float *current_out = out + batch_index * npoints * nsample * c;

      // 对于每个通道c和每个采样点npoints进行迭代
      for (int l = 0; l < c; ++l) { // 遍历每个通道
        for (int j = 0; j < npoints; ++j) { // 遍历每个采样点
          for (int k = 0; k < nsample; ++k) { // 对于每个样本点的nsample个邻居
            int ii = current_idx[j * nsample + k]; // 获取对应原始点的索引
            if(ii >= 0 && ii < n) { // 确保索引有效
              current_out[(l * npoints + j) * nsample + k] = current_points[l * n + ii];
            } else {
              // 如果索引无效，则可以设置一个默认值或者抛出异常等处理方式
              current_out[(l * npoints + j) * nsample + k] = 0.0f; // 这里简单地设置为0.0
            }
          }
        }
      }
    }
  }

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points);

void group_points_grad_kernel_cpu_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points){
    // 遍历每个batch
    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // 计算当前batch的偏移量
      const float *current_grad_out = grad_out + batch_index * npoints * nsample * c;
      const int *current_idx = idx + batch_index * npoints * nsample;
      float *current_grad_points = grad_points + batch_index * n * c;

      // 初始化梯度点数组为0，确保不会重复累加时出错
      for (int i = 0; i < n * c; ++i) {
        current_grad_points[i] = 0.0f;
      }

      // 对于每个通道c和每个采样点npoints进行迭代
      for (int l = 0; l < c; ++l) { // 遍历每个通道
        for (int j = 0; j < npoints; ++j) { // 遍历每个采样点
          for (int k = 0; k < nsample; ++k) { // 对于每个样本点的nsample个邻居
            int ii = current_idx[j * nsample + k]; // 获取对应原始点的索引
            if(ii >= 0 && ii < n) { // 确保索引有效
              // 累加梯度值到对应的grad_points位置
              current_grad_points[l * n + ii] += current_grad_out[(l * npoints + j) * nsample + k];
            }
            // 如果索引无效，则忽略该梯度贡献
          }
        }
      }
    }
  }

at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data<float>(),
                                idx.data<int>(), output.data<float>());
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    group_points_kernel_cpu_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data<float>(),
                                idx.data<int>(), output.data<float>());
  }

  return output;
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    group_points_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out.data<float>(), idx.data<int>(), output.data<float>());
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    group_points_grad_kernel_cpu_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out.data<float>(), idx.data<int>(), output.data<float>());
  }

  return output;
}
