from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'checkpoints/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# please prepare an image with person
results = inference_topdown(model, '../images/test_image_01.jpg')

print(results)
