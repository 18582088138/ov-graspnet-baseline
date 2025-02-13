import openvino as ov
from openvino import serialize
from openvino.runtime import Core
from openvino.frontend.onnx import OpExtension
from openvino.frontend import ConversionExtension, NodeContext

ov_extension_lib_path = "../ov_custom_op/build/libopenvino_operation_extension.so"


import onnx
import numpy as np
from onnx.helper import make_graph, make_model, make_tensor_value_info

from openvino.frontend import FrontEndManager

fem = FrontEndManager()

model_name = "ThreeInterpolate"

def create_onnx_model_extension():
    fps = onnx.helper.make_node(model_name, inputs=["features", "idx", "weight"], 
                                outputs=["sampled_indices"])
    # add = onnx.helper.make_node("CustomAdd", inputs=["x", "y"], outputs=["z"], domain="custom_domain")
    const_tensor = onnx.helper.make_tensor("const_tensor",
                                           onnx.TensorProto.FLOAT,
                                           (1,),
                                           [0.5])
    const_node = onnx.helper.make_node("Constant", [], outputs=["const_node"],
                                       value=const_tensor, name="const_node")
    mul = onnx.helper.make_node("Mul", inputs=["sampled_indices", "const_node"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("features", onnx.TensorProto.FLOAT, (1, 16, 128)),
        make_tensor_value_info("idx", onnx.TensorProto.INT32, (1, 64, 3)),
        make_tensor_value_info("weight", onnx.TensorProto.FLOAT, (1, 64, 3)),
    ]
    output_tensors = [make_tensor_value_info("out", onnx.TensorProto.FLOAT, (1, 16, 64))]
    graph = make_graph([fps, const_node, mul], "graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="ONNX Frontend")
    # model = shape_inference.infer_shapes(model)
    return model


onnx_model_with_extension_op = f"test_model/onnx_extension_op_{model_name}.onnx"
onnx.save_model(create_onnx_model_extension(), onnx_model_with_extension_op)
print(f"======== [Success] onnx extension sub model {onnx_model_with_extension_op} save ==========")

try:
    core = Core()
    core.add_extension(ov_extension_lib_path)
    ov_model = core.read_model(onnx_model_with_extension_op)
    ov_compiled_model = core.compile_model(ov_model, 'CPU')
    print(f"======== [Success] OpenVINO Compile {onnx_model_with_extension_op}=======")

except Exception as e:
    print(f"======== [Failed] OpenVINO Compile {onnx_model_with_extension_op} ,faile: {e} ==========")