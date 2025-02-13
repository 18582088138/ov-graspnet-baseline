// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
//! [op:common_include]

//! [op:header]
namespace TemplateExtension {

class ThreeInterpolate : public ov::op::Op {
public:
    OPENVINO_OP("ThreeInterpolate");

    ThreeInterpolate() = default;
    ThreeInterpolate(const ov::Output<ov::Node>& features, const ov::Output<ov::Node>& idx, const ov::Output<ov::Node>& weight);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
//! [op:header]

}  // namespace TemplateExtension