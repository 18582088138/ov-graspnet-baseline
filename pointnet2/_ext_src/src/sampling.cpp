// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_kernel_cpu_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out){
    // 遍历每个batch
    for (int i = 0; i < b; ++i) {
      // 对于每个通道c进行迭代
      for (int l = 0; l < c; ++l) {
        // 对于每个采样点m进行迭代
        for (int j = 0; j < npoints; ++j) {
          // 获取对应原始点的索引
          int a = idx[i * npoints + j];
          if(a >= 0 && a < n) { // 确保索引有效
            // 根据索引从points中提取对应的点值写入out
            out[(i * c + l) * npoints + j] = points[(i * c + l) * n + a];
          } else {
            // 如果索引无效，则可以设置一个默认值或者抛出异常等处理方式
            out[(i * c + l) * npoints + j] = 0.0f; // 这里简单地设置为0.0
          }
        }
      }
    }
  }
                          
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);
void gather_points_grad_kernel_cpu_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points){
    // 初始化梯度点数组为0，确保不会重复累加时出错
    for (int i = 0; i < b; ++i) {
      for (int l = 0; l < c; ++l) {
        for (int a = 0; a < n; ++a) {
          grad_points[(i * c + l) * n + a] = 0.0f;
        }
      }
    }

    // 遍历每个batch
    for (int i = 0; i < b; ++i) {
      // 对于每个通道c进行迭代
      for (int l = 0; l < c; ++l) {
        // 对于每个采样点m进行迭代
        for (int j = 0; j < npoints; ++j) {
          // 获取对应原始点的索引
          int a = idx[i * npoints + j];
          if(a >= 0 && a < n) { // 确保索引有效
            // 累加梯度值到对应的grad_points位置
            grad_points[(i * c + l) * n + a] += grad_out[(i * c + l) * npoints + j];
          }
          // 如果索引无效，则忽略该梯度贡献
        }
      }
    }
  }                               

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);                                         
void furthest_point_sampling_kernel_cpu_wrapper(int b, int n, int m,
                                            const float *dataset, int *idxs){
    if (m <= 0) return;

    for (int batch_index = 0; batch_index < b; ++batch_index) {
        // 每个batch中的起始位置
        const float *current_dataset = dataset + batch_index * n * 3;
        int *current_idxs = idxs + batch_index * m;

        // 初始化temp数组为最大值，表示初始时每个点到已选点集的距离未知或无限大
        std::vector<float> temp(n, std::numeric_limits<float>::max());

        // 初始化第一个点
        current_idxs[0] = 0;
        for (int j = 1; j < m; ++j) {
            int besti = 0;
            float best = -std::numeric_limits<float>::max();
            float x1 = current_dataset[current_idxs[j - 1] * 3 + 0];
            float y1 = current_dataset[current_idxs[j - 1] * 3 + 1];
            float z1 = current_dataset[current_idxs[j - 1] * 3 + 2];

            // 计算每个点的距离，并找到最远的点
            for (int k = 0; k < n; ++k) {
                float x2 = current_dataset[k * 3 + 0];
                float y2 = current_dataset[k * 3 + 1];
                float z2 = current_dataset[k * 3 + 2];
                float mag = x2 * x2 + y2 * y2 + z2 * z2;
                if (mag <= 1e-3) continue;

                float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
                float d2 = std::min(d, temp[k]);
                temp[k] = d2;
                if (d2 > best) {
                    best = d2;
                    besti = k;
                }
            }

            // 更新下一个选择的点
            current_idxs[j] = besti;
        }
    }
}

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data<float>(),
                                 idx.data<int>(), output.data<float>());
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    gather_points_kernel_cpu_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data<float>(),
                                 idx.data<int>(), output.data<float>());
  }

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
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
    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data<float>(),
                                      idx.data<int>(), output.data<float>());
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    gather_points_grad_kernel_cpu_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data<float>(),
                                      idx.data<int>(), output.data<float>());
  }

  return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data<float>(),
        tmp.data<float>(), output.data<int>());
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    furthest_point_sampling_kernel_cpu_wrapper(
        points.size(0), points.size(1), nsamples, points.data<float>(),
        output.data<int>());
  }

  return output;
}
