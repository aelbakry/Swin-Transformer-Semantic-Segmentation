from mmseg.models import build_segmentor
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

cfg = Config.fromfile('configs/swin/upernet_swin_base_patch4_window7_512x512_hubmap.py')

model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
chk_pth = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth'
checkpoint = load_checkpoint(model, chk_pth, map_location='cpu')

# print(model)
import torch
print(model(torch.zeros((2, 3, 256, 256))).shape)
