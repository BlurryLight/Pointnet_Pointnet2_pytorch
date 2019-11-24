import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    ##提取特征
    def __init__(self, global_feat=True, feature_transform=False, semseg = False):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d() if not semseg else STNkd(k=9)
        self.conv1 = torch.nn.Conv1d(3, 64, 1) if not semseg else torch.nn.Conv1d(9, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        #x=(32,3,2500)
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, cat_num=16,part_num=50):
        #这个有点魔改了吧？
        #既要预测是哪一类的，也要预测part num 
        super(PointNetDenseCls, self).__init__()
        self.cat_num = cat_num
        self.part_num = part_num
        self.stn = STN3d()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        # classification network
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, cat_num)
        self.dropout = nn.Dropout(p=0.3)
        self.bnc1 = nn.BatchNorm1d(256)
        self.bnc2 = nn.BatchNorm1d(256)
        # segmentation network

        self.convs1 = torch.nn.Conv1d(4928 + cat_num, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud,label):
        #(8,3,2500)
        batchsize,_ , n_pts = point_cloud.size()
        # point_cloud_transformed
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        point_cloud_transformed = torch.bmm(point_cloud, trans)
        point_cloud_transformed = point_cloud_transformed.transpose(2, 1)
        # MLP
        out1 = F.relu(self.bn1(self.conv1(point_cloud_transformed))) #3->64
        out2 = F.relu(self.bn2(self.conv2(out1))) # 64->128
        out3 = F.relu(self.bn3(self.conv3(out2))) # 128->128
        # net_transformed
        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)
        # MLP
        out4 = F.relu(self.bn4(self.conv4(net_transformed))) #128->512
        out5 = self.bn5(self.conv5(out4)) # 512->2048
        out_max = torch.max(out5, 2, keepdim=True)[0] # pooling 
        out_max = out_max.view(-1, 2048) # (batch_size , 2048)
        # classification network
        net = F.relu(self.bnc1(self.fc1(out_max))) # 2048->256
        net = F.relu(self.bnc2(self.dropout(self.fc2(net)))) # 256->256 * 0.3 dropou
        net = self.fc3(net) # [B,16] # 256->cat_num 
        # segmentation network
        #label [batch_size, 16] the 16 means what?
        # label是一个数组[0,1,0,0,0,0..],代表是第几个类
        # label的来源是通过utils的to_categorical实现的
        out_max = torch.cat([out_max,label],1)
        # print(out_max.shape)
        #out_max[8,2064]
        #expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, n_pts)
        expand = out_max.view(-1, 2048+label.shape[1], 1).repeat(1, 1, n_pts)
        #沿着列复制1024个值，代表1024个点
        #expand[8,2064,1024]
        # print(expand.shape)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        # concat[8,4944,1024]
        # 这里更激进一点，把out1,out2,out3,out4,out5都连起来了
        # out1(8,64,1024) out2(128) out3(128) out4(512) out5(2048)
        # 2064+64+128+128+512+2048 = 4944
        # 2064 = (2048 + cat_num)
        # print(concat.shape)
        net2 = F.relu(self.bns1(self.convs1(concat))) #(4944->256)
        net2 = F.relu(self.bns2(self.convs2(net2))) #256->256
        net2 = F.relu(self.bns3(self.convs3(net2))) #256->128
        net2 = self.convs4(net2) #256->partnum (8,partnum,1024)
        net2 = net2.transpose(2, 1).contiguous() # (8,1024,partnum)
        net2 = F.log_softmax(net2.view(-1, self.part_num), dim=-1)
        # net2.view以后，是(8192,partnum) 沿着行求log_softmax
        net2 = net2.view(batchsize, n_pts, self.part_num) # [B, N 50]
        # 重新变形回(8,1024,partnum)

        return net, net2, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss

class PointNetLoss(torch.nn.Module):
    def __init__(self, weight=1,mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.weight = weight

    def forward(self, labels_pred, label, seg_pred,seg, trans_feat):
        seg_loss = F.nll_loss(seg_pred, seg)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        label_loss = F.nll_loss(labels_pred, label)

        loss = self.weight * seg_loss + (1-self.weight) * label_loss + mat_diff_loss * self.mat_diff_loss_scale
        # 不考虑label loss 那你预测个***
        return loss, seg_loss, label_loss


class PointNetSeg(nn.Module):
    def __init__(self,num_class,feature_transform=False, semseg = False):
        super(PointNetSeg, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False,feature_transform=feature_transform, semseg = semseg)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat



if __name__ == '__main__':
    point = torch.randn(8,3,1024)
    label = torch.randn(8,1)
    model = PointNetDenseCls(1,3)
    net, net2, trans_feat = model(point,label)
    print('net',net.shape)
    print('net2',net2.shape)
    print('trans_feat',trans_feat.shape)
