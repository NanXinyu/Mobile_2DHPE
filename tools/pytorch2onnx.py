import onnx
import torch
import numpy
import onnxruntime as ort
from torchsummary import summary
import _init_paths
from model import get_pose_net
from config import cfg

class Transformer():
    def __init__(self):
        super(Transformer, self).__init__()
        self.modelpath = '../output/LiteModel.pth'

    def _make_model(self):
        # prepare network
        model = get_pose_net(cfg, False)
   
        model.load_state_dict(torch.load(self.modelpath))
        single_pytorch_model = model#.module
        single_pytorch_model.eval()
        self.model = single_pytorch_model


dummy_input = torch.randn(1, 3, 256, 192, device='cuda')

# modelpath as definite path
transformer = Transformer()
transformer._make_model()

single_pytorch_model = transformer.model

summary(single_pytorch_model.cuda(), (3, 256, 192))

ONNX_PATH="../output/LiteModel.onnx"

torch.onnx.export(
    model=single_pytorch_model,
    args=dummy_input,
    f=ONNX_PATH, # where should it be saved
    verbose=False,
    export_params=True,
    do_constant_folding=False,  # fold constant values for optimization
    # do_constant_folding=True,   # fold constant values for optimization
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)

pytorch_result_x, pytorch_result_y = single_pytorch_model(dummy_input)
pytorch_result_x = pytorch_result_x.cpu().detach().numpy()
pytorch_result_y = pytorch_result_y.cpu().detach().numpy()
print("pytorch_model output x {}".format(pytorch_result_x.shape), pytorch_result_x)
print("pytorch_model output y {}".format(pytorch_result_y.shape), pytorch_result_y)

ort_session = ort.InferenceSession(ONNX_PATH)
output_x, output_y= ort_session.run(None, {'input': dummy_input.cpu().numpy()})
output_x = numpy.array(output_x[0])
output_y = numpy.array(output_y[0])
print("onnx_model ouput x size{}".format(output_x.shape), output_x)
print("onnx_model ouput y size{}".format(output_y.shape), output_y)
print("difference x", numpy.linalg.norm(pytorch_result_x-output_x))
print("difference y", numpy.linalg.norm(pytorch_result_y-output_y))

