// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_operation.hpp"

using namespace TemplateExtension;

//! [op:ctor]
GatherOperation::GatherOperation(const ov::Output<ov::Node>& features, const ov::Output<ov::Node>& idx) : Op({features, idx}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void GatherOperation::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    /*
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor

    idx : torch.Tensor
        (B, npoint) tensor of the features to gather

    Returns
    -------
    torch.Tensor
        (B, C, npoint) tensor
    */
    const auto& features_input = input(0);
    const auto& idx_input = input(1);

    auto features_shape = features_input.get_partial_shape();
    auto idx_shape = idx_input.get_partial_shape();
    ov::PartialShape output_shape = {features_shape[0], features_shape[1], idx_shape[1]};
    set_output_type(0, features_input.get_element_type(), output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> GatherOperation::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<GatherOperation>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool GatherOperation::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool GatherOperation::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* features = inputs[0].data<const float>();
    const int* idx = inputs[1].data<const int>();

    int b = inputs[0].get_shape()[0]; // batch size
    int c = inputs[0].get_shape()[1]; // channels
    int n = inputs[0].get_shape()[2]; // number of points
    int npoints = inputs[1].get_shape()[1]; // number of points to gather

    ov::PartialShape output_shape = {b, c, npoints};
    outputs[0].set_shape(output_shape.to_shape());
    auto* out_tensor = outputs[0].data<float>();

    for (int i = 0; i < b; ++i) {
        // 对于每个通道c进行迭代
        for (int l = 0; l < c; ++l) {
            // 对于每个采样点m进行迭代
            for (int j = 0; j < npoints; ++j) {
                // 获取对应原始点的索引
                int a = idx[i * npoints + j];
                if(a >= 0 && a < n) { // 确保索引有效
                    // 根据索引从points中提取对应的点值写入out
                    out_tensor[(i * c + l) * npoints + j] = features[(i * c + l) * n + a];
                } else {
                    // 如果索引无效，则可以设置一个默认值或者抛出异常等处理方式
                    out_tensor[(i * c + l) * npoints + j] = 0.0f; // 这里简单地设置为0.0
                }
            }
        }
    }
    // out.set_shape(in.get_shape());
    // memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool GatherOperation::has_evaluate() const {
    return true;
}
//! [op:evaluate]