// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "three_interpolate.hpp"

using namespace TemplateExtension;

//! [op:ctor]
ThreeInterpolate::ThreeInterpolate(const ov::Output<ov::Node>& features, const ov::Output<ov::Node>& idx, const ov::Output<ov::Node>& weight) : Op({features, idx, weight}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void ThreeInterpolate::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    /*
    Parameters
    ----------
    features : torch.Tensor
        (B, c, m) Features descriptors to be interpolated from
    idx : torch.Tensor
        (B, n, 3) three nearest neighbors of the target features in features
    weight : torch.Tensor
        (B, n, 3) weights

    Returns
    -------
    torch.Tensor
        (B, c, n) tensor of the interpolated features
    */
   const auto& features = input(0);
   const auto& idx = input(1);
   const auto& weight = input(2);

    auto features_shape = features.get_partial_shape();
    auto idx_shape = idx.get_partial_shape();
    ov::PartialShape output_shape = {features_shape[0], features_shape[1], idx_shape[1]};
    set_output_type(0, ov::element::f32, output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> ThreeInterpolate::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<ThreeInterpolate>(new_args.at(0), new_args.at(1), new_args.at(2));
}
//! [op:copy]

//! [op:visit_attributes]
bool ThreeInterpolate::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool ThreeInterpolate::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* features = inputs[0].data<const float>();
    const int* idx = inputs[1].data<const int>();
    const float* weight = inputs[2].data<const float>();

    int b = inputs[0].get_shape()[0];
    int c = inputs[0].get_shape()[1];
    int m = inputs[0].get_shape()[2];
    int n = inputs[1].get_shape()[1];

    ov::PartialShape output_shape = {b, c, n};
    outputs[0].set_shape(output_shape.to_shape());

    auto& out_tensor = outputs[0];
    float *out_data = out_tensor.data<float>();

    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // 计算当前batch的偏移量
      const float *current_points = features + batch_index * m * c;
      const int *current_idx = idx + batch_index * n * 3;
      const float *current_weight = weight + batch_index * n * 3;
      float *current_out = out_data + batch_index * n * c;

      // 对于每个通道c和每个点n进行迭代
      for (int l = 0; l < c; ++l) { // 遍历每个通道
        for (int j = 0; j < n; ++j) { // 遍历每个点
          float w1 = current_weight[j * 3 + 0];
          float w2 = current_weight[j * 3 + 1];
          float w3 = current_weight[j * 3 + 2];

          int i1 = current_idx[j * 3 + 0];
          int i2 = current_idx[j * 3 + 1];
          int i3 = current_idx[j * 3 + 2];

          // 确保索引有效
          if(i1 >= 0 && i1 < m && i2 >= 0 && i2 < m && i3 >= 0 && i3 < m) {
            current_out[l * n + j] = current_points[l * m + i1] * w1 +
                                    current_points[l * m + i2] * w2 +
                                    current_points[l * m + i3] * w3;
          } else {
            // 如果索引无效，则可以设置一个默认值或者抛出异常等处理方式
            current_out[l * n + j] = 0.0f; // 这里简单地设置为0.0
          }
        }
      }
    }

    // out.set_shape(in.get_shape());
    // memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool ThreeInterpolate::has_evaluate() const {
    return true;
}
//! [op:evaluate]