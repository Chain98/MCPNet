import os
import argparse
import numpy as np
from PIL import Image
from model.models import Net
import torch
import torchvision.transforms as tfs
import torchvision.utils as vutils
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

abs = os.getcwd() + '/'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='haze')
parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Test imgs folder')
opt = parser.parse_args()
dataset = opt.task
img_dir = abs + opt.test_imgs + '/'
output_dir = abs + f'pred_MCP_{dataset}/'
print("pred_dir:", output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_dir = abs + f'trained_models/{dataset}_train_MCP.tar'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

device_ids = [0]

ckp = torch.load(model_dir, map_location=device)
net = Net()
net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
net.load_state_dict(ckp['state_dict'])
optimizer = torch.optim.Adam(net.parameters())
optimizer.load_state_dict(ckp['optimizer'])
net.eval()

def norm(x):
    return (x - 0.5) / 0.5

def denorm(x):
    return (x + 1) / 2

for im in os.listdir(img_dir):
    image_path = os.path.join(img_dir, im)
    haze = np.array(Image.open(image_path).convert('RGB')) / 255

    with torch.no_grad():
        haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :]).cuda()
        haze = norm(haze)
        x1, x2, pred = net(haze)
        out = denorm(pred)

    # 保存图像
    output_image_path = os.path.join(output_dir, im.split('.')[0] + '_MCP.png')
    vutils.save_image(out, output_image_path)

    # 显示图像
    out_image = tfs.ToPILImage()(out.squeeze(0).cpu())
    plt.imshow(out_image)
    plt.title('MCP Prediction')
    plt.show()
