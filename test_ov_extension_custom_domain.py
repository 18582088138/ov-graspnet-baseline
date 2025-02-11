import openvino as ov
from openvino.runtime import Core
from openvino import serialize

ov_extension_lib_path = "ov_custom_op/build/libopenvino_operation_extension.so"
onnx_model_extension_with_custom_domain = "model_extension_custom_domain.onnx"

core = Core()
ov_extension_lib_path = 'ov_custom_op/build/libopenvino_operation_extension.so'
core.add_extension(ov_extension_lib_path)
ov_model = ov.read_model(onnx_model_extension_with_custom_domain)
ov_compiled_model = core.compile_model(ov_model, 'CPU')

print("=======Load model_extension_custom_domain success=======")