import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import torch

from lib.options import BaseOptions
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')
        print(cuda)

        # create net
        if opt.anchor:
            netG = AnchorUdfNet(opt, projection_mode).to(device=cuda)
        else:
            netG = UdfNet(opt, projection_mode).to(device=cuda)

        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            if opt.mgpu:
                state_dict = torch.load(opt.load_netG_checkpoint_path, map_location=cuda)
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                netG.load_state_dict(new_state_dict)
            else:
                netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG

    def load_image(self, image_path, mask_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        img_ids = image_path.split('/')[-2].split('_')
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        # Mask
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        return {
            'name': img_ids[0]+'_'+img_ids[-1]+'_'+img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data):
        opt = self.opt
        self.netG.eval()
        save_path = '%s/%s/%s.obj' % (opt.results_path, opt.name, data['name'])
        gen_mesh_udf(opt, self.netG, self.cuda, data, save_path, opt.num_steps, opt.num_gen_points)


if __name__ == '__main__':
    evaluator = Evaluator(opt)
    yaw_list = [0]

    for vid in yaw_list:
        image_path = os.path.join(opt.dataroot, 'RENDER', opt.test_folder_path, '%d_%d_%02d.jpg' % (vid, 0, 0))
        mask_path = os.path.join(opt.dataroot, 'MASK', opt.test_folder_path, '%d_%d_%02d.png' % (vid, 0, 0))
        print(image_path, mask_path)

        data = evaluator.load_image(image_path, mask_path)
        evaluator.eval(data)
