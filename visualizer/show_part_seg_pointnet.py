import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import torch
import torch.nn.parallel
import torch.utils.data
from utils import to_categorical
from collections import defaultdict
from torch.autograd import Variable
from data_utils.MyShapeNetDataLoader import PartNormalDataset
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from utils import test_partseg
from tqdm import tqdm
from model.pointnet import PointNetDenseCls,PointNetLoss
import matplotlib.pyplot as plt
import numpy as np
from show3d_balls import showpoints

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--class_choice', type=str, default=None, help='class choice',nargs='+')

opt = parser.parse_args()
print(opt)
class_choice = opt.class_choice
if class_choice is None:
    class_choice = []
d = PartNormalDataset(root_path = "/mnt/Gpan/QTC++/githubBase/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/",split='test',class_choice=class_choice)
print(len(d))

idx = opt.idx
point,cls,seg,_ = d[idx]
# print(point)
# print(seg)
# print(cls)
# print(len(point),len(seg))

#设置cmap颜色
cmap = plt.cm.get_cmap("hsv",50)
cmap = np.array([cmap(i) for i in range(50)])[:,:3]
gt = cmap[seg -1,:]
# showpoints(point,gt)

state_dict = torch.load(opt.model)

num_classes = len(d.cat)
num_part = d.part_num_sum

classifier = PointNetDenseCls(cat_num=num_classes,part_num=num_part)
classifier.load_state_dict(state_dict)
classifier.eval()
point_np = point
point = torch.from_numpy(point)
cls = torch.from_numpy(cls)
seg = torch.from_numpy(seg)
point, cls, seg= Variable(point.float()),Variable(cls.long()),  Variable(seg.long())
# print(point.shape)
point = point.transpose(1,0).contiguous()
point = Variable(point.view(1,point.size()[0],point.size()[1]))
print(point.shape)

if torch.cuda.is_available():
    print("cuda!!")
    classifier.cuda()
    point,cls,seg = point.cuda(),cls.cuda(),seg.cuda()

print(cls)
cls_pred,seg_pred,_= classifier(point,to_categorical(cls,num_classes))
seg_pred = seg_pred.contiguous().view(-1,num_part)
# print(seg_pred.shape)
print(cls_pred)
print(cls_pred.shape)
print(cls_pred.max(0)[1])
print(cls_pred.max(1)[1])
pred_choice = seg_pred.data.max(1)[1]
print(max(pred_choice),min(pred_choice))
correct = pred_choice.eq(seg.data).cpu().sum()
print("coreect",correct)
print("acc",correct.item()/2500)
print(pred_choice.shape)
dif = (pred_choice == seg).cpu().numpy().astype(int)
print(dif)
dif_color = cmap[dif,:]
# pred_color = cmap[pred_choice.cpu().numpy(),:]

showpoints(point_np,dif_color)
