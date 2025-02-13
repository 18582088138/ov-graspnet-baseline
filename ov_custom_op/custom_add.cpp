// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_add.hpp"

using namespace TemplateExtension;

//! [op:ctor]
CustomAdd::CustomAdd(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& y) : Op({x, y}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomAdd::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomAdd::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");

    return std::make_shared<CustomAdd>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomAdd::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CustomAdd::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& in = inputs[0];
    auto& out = outputs[0];
    if (out.data() == in.data())  // Nothing to do
        return true;
    out.set_shape(in.get_shape());
    memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool CustomAdd::has_evaluate() const {
    return true;
}
//! [op:evaluate]