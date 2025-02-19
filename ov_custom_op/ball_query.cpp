// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ball_query.hpp"

using namespace TemplateExtension;

//! [op:ctor]
BallQuery::BallQuery(const ov::Output<ov::Node>& radius, const ov::Output<ov::Node>& nsample, 
                    const ov::Output<ov::Node>& xyz, const ov::Output<ov::Node>& new_xyz) : Op({radius, nsample, xyz, new_xyz}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void BallQuery::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    /*
    Parameters
    ----------
    radius : float
        radius of the balls
    nsample : int
        maximum number of features in the balls
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the ball query

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    */
    const auto& radius = input(0);
    const auto& nsample = input(1);
    const auto& xyz = input(2);
    const auto& new_xyz = input(3);
    auto new_xyz_shape = new_xyz.get_partial_shape();
    ov::PartialShape output_shape = {new_xyz_shape[0], new_xyz_shape[1], -1}; // 64 as template value. The value of output shape needs to be updated during inference.

    set_output_type(0, ov::element::i32, output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> BallQuery::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<BallQuery>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}
//! [op:copy]

//! [op:visit_attributes]
bool BallQuery::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool BallQuery::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float radius = *inputs[0].data<const float>();
    const int nsample = *inputs[1].data<const int>();
    const float* xyz = inputs[2].data<const float>();
    const float* new_xyz = inputs[3].data<const float>();

    int b = inputs[2].get_shape()[0]; // batch size
    int n = inputs[2].get_shape()[1]; // number of points in xyz
    int npoint = inputs[3].get_shape()[1]; // number of points in new_xy
    // int m = inputs[3].get_shape()[1];

    ov::PartialShape output_shape = {b, npoint, nsample};
    outputs[0].set_shape(output_shape.to_shape());
    auto& out_tensor = outputs[0];
    int *current_idx = out_tensor.data<int>();

    float radius2 = radius * radius;

    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // 每个batch中的起始位置
      const float *current_xyz = xyz + batch_index * n * 3;
      const float *current_new_xyz = new_xyz + batch_index * npoint * 3;
      int *current_batch_idx = current_idx + batch_index * npoint * nsample;

      for (int j = 0; j < npoint; ++j) {
        float new_x = current_new_xyz[j * 3 + 0];
        float new_y = current_new_xyz[j * 3 + 1];
        float new_z = current_new_xyz[j * 3 + 2];
        int cnt = 0;

        // 遍历所有原始点以找到在指定半径内的点
        for (int k = 0; k < n && cnt < nsample; ++k) {
          float x = current_xyz[k * 3 + 0];
          float y = current_xyz[k * 3 + 1];
          float z = current_xyz[k * 3 + 2];
          float d2 = (new_x - x) * (new_x - x) +
                    (new_y - y) * (new_y - y) +
                    (new_z - z) * (new_z - z);

          if (d2 < radius2) {
            if (cnt == 0) {
              // 初始化索引数组，如果找不到足够的邻居，则重复最后一个有效的邻居索引
              for (int l = 0; l < nsample; ++l) {
                current_batch_idx[j * nsample + l] = k;
              }
            }
            current_batch_idx[j * nsample + cnt] = k;
            ++cnt;
          }
        }

        // 如果找到的点少于nsample，则填充剩余索引为最后一个有效索引或-1
        // while (cnt < nsample) {
        //   current_batch_idx[j * nsample + cnt] = (cnt == 0) ? -1 : current_batch_idx[j * nsample + cnt - 1];
        //   ++cnt;
        // }
      }
    }
    // out.set_shape(in.get_shape());
    // memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool BallQuery::has_evaluate() const {
    return true;
}
//! [op:evaluate]