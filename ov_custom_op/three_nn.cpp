// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "three_nn.hpp"

using namespace TemplateExtension;

//! [op:ctor]
ThreeNN::ThreeNN(const ov::Output<ov::Node>& unknown, const ov::Output<ov::Node>& known) : Op({unknown, known}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void ThreeNN::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    /*
    Parameters
    ----------
    unknown : torch.Tensor
        (B, n, 3) tensor of known features
    known : torch.Tensor
        (B, m, 3) tensor of unknown features

    Returns
    -------
    combined : 
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
    */
    const auto& unknown = input(0);
    const auto& known = input(1);
    auto unknown_shape = unknown.get_partial_shape();
    ov::PartialShape output_dist_shape = {unknown_shape[0], unknown_shape[1], 3}; 
    ov::PartialShape output_idx_shape = {unknown_shape[0], unknown_shape[1], 3}; 
    set_output_type(0, ov::element::f32, output_dist_shape);
    set_output_type(1, ov::element::i32, output_idx_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> ThreeNN::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<ThreeNN>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool ThreeNN::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool ThreeNN::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* unknown = inputs[0].data<const float>();
    const float* known = inputs[1].data<const float>();

    int b = inputs[0].get_shape()[0]; // batch size
    int n = inputs[0].get_shape()[1]; // number of points in xyz
    int m = inputs[1].get_shape()[1];

    auto& out_dist_tensor = outputs[0];
    float *out_dist_data = out_dist_tensor.data<float>();

    auto& out_idx_tensor = outputs[1];
    int *out_idx_data = out_idx_tensor.data<int>();

    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // 计算当前batch的偏移量
      const float *current_unknown = unknown + batch_index * n * 3;
      const float *current_known = known + batch_index * m * 3;
      float *current_dist = out_dist_data + batch_index * n * 3;
      int *current_idx = out_idx_data + batch_index * n * 3;

      // 对于每个未知点进行迭代
      for (int j = 0; j < n; ++j) {
        float ux = current_unknown[j * 3 + 0];
        float uy = current_unknown[j * 3 + 1];
        float uz = current_unknown[j * 3 + 2];

        double best1 = std::numeric_limits<double>::max();
        double best2 = std::numeric_limits<double>::max();
        double best3 = std::numeric_limits<double>::max();
        int besti1 = -1, besti2 = -1, besti3 = -1;

        // 遍历所有已知点以找到最近的三个点
        for (int k = 0; k < m; ++k) {
          float x = current_known[k * 3 + 0];
          float y = current_known[k * 3 + 1];
          float z = current_known[k * 3 + 2];
          float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

          if (d < best1) {
            best3 = best2;
            besti3 = besti2;
            best2 = best1;
            besti2 = besti1;
            best1 = d;
            besti1 = k;
          } else if (d < best2) {
            best3 = best2;
            besti3 = besti2;
            best2 = d;
            besti2 = k;
          } else if (d < best3) {
            best3 = d;
            besti3 = k;
          }
        }

        // 更新结果数组
        current_dist[j * 3 + 0] = static_cast<float>(best1);
        current_dist[j * 3 + 1] = static_cast<float>(best2);
        current_dist[j * 3 + 2] = static_cast<float>(best3);

        current_idx[j * 3 + 0] = besti1;
        current_idx[j * 3 + 1] = besti2;
        current_idx[j * 3 + 2] = besti3;
      }
    }
    // out.set_shape(in.get_shape());
    // memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool ThreeNN::has_evaluate() const {
    return true;
}
//! [op:evaluate]