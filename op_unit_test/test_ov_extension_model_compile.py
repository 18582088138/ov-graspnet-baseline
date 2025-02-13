import openvino as ov
import onnx
from openvino.runtime import Core
from openvino import serialize

model_name = "CylinderQuery"
onnx_model_extension_with_custom_domain = "onnx_extension_CylinderQuery.onnx"
ov_extension_lib_path = 'ov_custom_op/build/libopenvino_operation_extension.so'

core = Core()
core.add_extension(ov_extension_lib_path)
print("=======Load extension success=======")
ov_model = core.read_model(onnx_model_extension_with_custom_domain)
print("=======Read model success=======")
ov_compiled_model = core.compile_model(ov_model, 'CPU')
print("=======Compile model success=======")

# ov_grasp_generator = ov.convert_model(grasp_generator, input=grasp_generator_input_name, example_input=grasp_generator_example_input, extension=ov_extension_lib_path)
ov.save_model(ov_model, 'IR_model/ov_sub_CylinderQuery.xml')
print("=======Save model success=======")

print("=======Load model_extension_custom_domain success=======")

