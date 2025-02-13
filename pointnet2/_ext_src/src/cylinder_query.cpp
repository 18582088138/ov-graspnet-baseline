// Author: chenxi-wang

#include "cylinder_query.h"
#include "utils.h"

void query_cylinder_point_kernel_wrapper(int b, int n, int m, float radius, float hmin, float hmax,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *rot, int *idx);

void query_cylinder_point_kernel_cpu_wrapper(int b, int n, int m, float radius, float hmin, float hmax,
                                             int nsample, const float *new_xyz,
                                             const float *xyz, const float *rot, int *idx){
    // 计算半径平方值
    float radius2 = radius * radius;

    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // 每个batch中的起始位置
      const float *current_xyz = xyz + batch_index * n * 3;
      const float *current_new_xyz = new_xyz + batch_index * m * 3;
      const float *current_rot = rot + batch_index * m * 9;
      int *current_idx = idx + batch_index * m * nsample;

      for (int j = 0; j < m; ++j) {
        // 获取当前点坐标和旋转矩阵
        float new_x = current_new_xyz[j * 3 + 0];
        float new_y = current_new_xyz[j * 3 + 1];
        float new_z = current_new_xyz[j * 3 + 2];
        float r[9] = {current_rot[j * 9 + 0], current_rot[j * 9 + 1], current_rot[j * 9 + 2],
                      current_rot[j * 9 + 3], current_rot[j * 9 + 4], current_rot[j * 9 + 5],
                      current_rot[j * 9 + 6], current_rot[j * 9 + 7], current_rot[j * 9 + 8]};

        int cnt = 0;
        for (int k = 0; k < n && cnt < nsample; ++k) {
          // 计算点相对于新点的位置，并应用旋转
          float x = current_xyz[k * 3 + 0] - new_x;
          float y = current_xyz[k * 3 + 1] - new_y;
          float z = current_xyz[k * 3 + 2] - new_z;
          float x_rot = r[0] * x + r[3] * y + r[6] * z;
          float y_rot = r[1] * x + r[4] * y + r[7] * z;
          float z_rot = r[2] * x + r[5] * y + r[8] * z;

          // 判断是否在圆柱体内
          float d2 = y_rot * y_rot + z_rot * z_rot;
          if (d2 < radius2 && x_rot > hmin && x_rot < hmax) {
            current_idx[j * nsample + cnt] = k;
            ++cnt;
          }
        }

        // 如果找到的点少于nsample，则填充剩余索引为最后一个有效索引或-1
        while (cnt < nsample) {
          current_idx[j * nsample + cnt] = (cnt == 0) ? -1 : current_idx[j * nsample + cnt - 1];
          ++cnt;
        }
      }
    }
  }

at::Tensor cylinder_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor rot, 
                      const float radius, const float hmin, const float hmax,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(rot);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(rot);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(rot);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    query_cylinder_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, hmin, hmax, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), rot.data<float>(), idx.data<int>());
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    query_cylinder_point_kernel_cpu_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                            radius, hmin, hmax, nsample, new_xyz.data<float>(),
                                            xyz.data<float>(), rot.data<float>(), idx.data<int>());
  }

  return idx;
}
