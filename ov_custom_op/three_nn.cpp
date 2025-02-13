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
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    set_output_type(1, get_input_element_type(0), get_input_partial_shape(0));
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
    const auto& in = inputs[0];
    auto& out = outputs[0];
    if (out.data() == in.data())  // Nothing to do
        return true;
    out.set_shape(in.get_shape());
    memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool ThreeNN::has_evaluate() const {
    return false;
}
//! [op:evaluate]