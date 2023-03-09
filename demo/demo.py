import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from config import cfg
from model import get_pose_net
from trans import crop, transform_preds, box_to_center_scale, vis_keypoints

cudnn.benchmark = True

# CoCo joint set
joint_num = 17
joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
flip_pairs = ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) )
skeleton =((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5,6), (5,7), (6,8), (7,9), (8, 10), (1,2), (0,1),(0,2),(1,3),(2,4),(3,5),(4,6))
model = get_pose_net(cfg, is_train=False)
if cfg.MODEL.NAME == 'Gaint':
    model_path = './assets/GaintModel.pth'
elif cfg.MODEL.NAME == 'Lite':
    model_path = './assets/LiteModel.pth'

model.load_state_dict(torch.load(model_path), strict = False)
model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
model.eval()

# prepare input image
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.ToTensor(), 
                                normalize,])

img_path = './assets/input.jpg'

original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
bbox_list = [
[139.41, 102.25, 222.39, 241.57],\
[287.17, 61.52, 74.88, 165.61],\
[540.04, 48.81, 99.96, 223.36],\
[372.58, 170.84, 266.63, 217.19],\
[0.5, 43.74, 90.1, 220.09]] # xmin, ymin, width, height

person_num = len(bbox_list)
# for each cropped and resized human image, forward it to PoseNet
output_pose_2d_list = []
model_input = cfg.MODEL.IMAGE_SIZE
for n in range(person_num):
    center, scale = box_to_center_scale(np.array(bbox_list[n]), model_input)
    img = crop(original_img, center, scale, model_input)

    ## get crop 
    # cv2.imwrite(f'./output/bbox_2d_{n}.jpg', img)
    
    img = transform(img).cuda().unsqueeze(0) 
    
    ## get normalized input
    # bbox_img = img[0].permute(1, 2, 0) 
    # bbox_img = bbox_img.cpu().numpy()
    # cv2.imwrite(f'./output/bbox_2d_norm_{n}.jpg', bbox_img * 255)

    # forward
    with torch.no_grad():
        pose_x, pose_y = model(img) 
    
    max_val_x, preds_x = pose_x.max(2)
    max_val_y, preds_y = pose_y.max(2)
    output = torch.ones([joint_num, 2])
    output[:, 0] = torch.squeeze(torch.true_divide(preds_x, cfg.MODEL.REDUCTION_RATIO))
    output[:, 1] = torch.squeeze(torch.true_divide(preds_y, cfg.MODEL.REDUCTION_RATIO))
   
    output = output.cpu().numpy()
    preds = transform_preds(output, center, scale, model_input)
    output_pose_2d_list.append(preds[:,:2].copy())
    
# visualize 2d poses
vis_img = original_img.copy()
for n in range(person_num):
    vis_kps = np.zeros((3,joint_num))
    vis_kps[0,:] = output_pose_2d_list[n][:,0]
    vis_kps[1,:] = output_pose_2d_list[n][:,1]
    vis_kps[2,:] = 1
    vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
    print(vis_img.shape)
cv2.imwrite('./output/test_2d.jpg', vis_img)

