import sys
import torch.onnx
from vision.ssd.config.fd_config import define_img_size

input_img_size = 320  
define_img_size(input_img_size)
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd

net_type = "RFB" 

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    net = create_mb_tiny_fd(2, is_test=True)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    net = create_Mb_Tiny_RFB_fd(2, is_test=True)

else:
    print("unsupport network type.")
    sys.exit(1)
net.load(model_path)
net.eval()
net.to("cuda")

model_name = model_path.split("/")[-1].split(".")[0]
model_path = f"models/onnx/RFB-test.onnx"

dummy_input = torch.randn(1, 3, 240, 320).to("cuda")
torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])
