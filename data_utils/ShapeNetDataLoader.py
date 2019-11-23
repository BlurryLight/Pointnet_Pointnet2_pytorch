# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

class PartNormalDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=True, jitter=False):
        self.npoints = npoints
        self.root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        # self.root = '/mnt/Gpan/QTC++/githubBase/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize
        self.jitter = jitter

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            print("train_ids",len(train_ids))
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            print("val_ids",len(val_ids))
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            print("test_ids",len(test_ids))
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item],'points')
            dir_seg = os.path.join(self.root,self.cat[item],'points_label')
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns[0]))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                # print(token)
                self.meta[item].append((os.path.join(dir_point, token + '.pts'),os.path.join(dir_seg,token+'.seg')))
        print("meta",self.meta.keys(),list(self.meta.values())[0][0])
        

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
        print("datapath", self.datapath[0])
        #datapath ('Airplane', '/mnt/Gpan/QTC++/githubBase/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points/1021a0914a7207aff927ed529ad90a11.txt')
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        for cat in sorted(self.seg_classes.keys()):
            print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1][0]).astype(np.float32)
            # normal = data[:, 3:6]
            # 这里要注意 去除了normal
            normal = np.zeros(point_set.shape)
            offset = self.seg_classes[cat][0]
            seg = np.loadtxt(fn[1][1]).astype(np.int32) + offset - 1
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)
                # self.cache[index] = (point_set, seg, cls)
        if self.normalize:
            point_set = pc_normalize(point_set)
        if self.jitter:
            jitter_point_cloud(point_set)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]
        # added for special shapenet




        return point_set,cls, seg, normal
        # return point_set,cls, seg

    def get_classname(self,index):
        fn = self.datapath[index]
        cat = self.datapath[index][0]
        return cat


    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    d = PartNormalDataset()
    print(len(d))
    ps,cls,seg,nom= d[1]
    print(len(ps),len(seg))
    print(d.get_classname(1))
    print(ps[0:4],seg[0:4],cls)

    d = PartNormalDataset(split='test')
    print(len(d))
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    for i in range(len(d)):
        ps,cls,seg,nom= d[i]
        part_list = seg_classes[d.get_classname(i)]
        if (min(seg)<min(part_list)):
            print('min')
            print(min(seg),min(part_list),d.get_classname(i))
        if (max(seg)>max(part_list)):
            print('max')
            print(max(seg),max(part_list),d.get_classname(i))
