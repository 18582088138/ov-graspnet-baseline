#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "identity.hpp"
#include "furthest_point_sampling.hpp"
#include "cylinder_query.hpp"
#include "grouping_operation.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({

        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::Identity>>(),
        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::Identity>>(),

        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::FurthestPointSampling>>(),
        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::FurthestPointSampling>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CylinderQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CylinderQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GroupingOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GroupingOperation>>(),
        
    }));
//! [ov_extension:entry_point]
// clang-format on