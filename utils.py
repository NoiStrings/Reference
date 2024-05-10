import torch
import torch.nn.functional as F
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
from einops import rearrange
import imageio
from func_pfm import *


class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        files = os.listdir(cfg.trainset_dir)
        files = sorted(files)
        self.trainset_dir = cfg.trainset_dir
        self.files = files
        self.angRes = cfg.angRes
        self.patchsize = cfg.patchsize
        """ We use totally 16 LF images (0 to 15) for training. 
            Since some images (4,6,15) have a reflection region,
                we decrease the occurrence frequency of them. """
                # 减少了训练集中存在反射面的图片
        scene_idx = []
        # 存储不同场景的索引
        for i in range(40):
            scene_idx = np.append(scene_idx, [0,1,2,3,4,5,6,7,8,9,10,11,12])
            # 去除4、6、15号场景的索引，占75%
        # scene_idx中共有1375个场景索引，存在大量重复
        # 通过后续的数据增强提高训练数据的多样性
        self.scene_idx = scene_idx.astype('int')
        self.item_num = len(self.scene_idx)

    def __getitem__(self, index):
        # index: 场景索引，每个场景包含9*9个不同视角
        scene_id = self.scene_idx[index]
        scene_name = self.files[scene_id]
        lf = np.zeros(shape=(9, 9, 512, 512, 3), dtype=int)
        # 角度分辨率9*9，空间分辨率512*512，3通道数（RGB）

        """ Read inputs """
        for i in range(81):
            temp = imageio.imread(self.trainset_dir + scene_name + '/input_Cam0%.2d.png' % i)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
            # 将81个视角按顺序读入lf中

        dispGT = np.zeros(shape=(512, 512, 2), dtype=float)
        dispGT[:, :, 0] = np.float32(read_pfm(self.trainset_dir + scene_name + '/gt_disp_lowres.pfm'))
        # dispGT[:, :, 0]用于储存实际视差图
        mask_rgb = imageio.imread(self.trainset_dir + scene_name + '/valid_mask.png')
        # 读取实际视差的掩膜
        dispGT[:, :, 1] = np.float32(mask_rgb[:, :, 1] > 0)
        # dispGT[:, :, 1]用于储存实际视差的掩膜
        # 判断mask_rgb中各像素的绿色通道是否 > 0，大于0代表该像素的视差值有效
        dispGT = dispGT.astype('float32')

        """ Data Augmentation """
        lf = illuminance_augmentation((1/255) * lf.astype('float32'))
        #lf = viewpoint_augmentation(lf, self.angRes)
        lf, dispGT, scale = scale_augmentation(lf, dispGT, self.patchsize)
        if scale == 1:
            lf, dispGT, refocus_flag = refocus_augmentation(lf, dispGT)
        else:
            refocus_flag = 0
        
        sum_diff = 0
        glass_region = False
        while (sum_diff < 0.01 or glass_region == True):
            lf_crop, dispGT_crop = random_crop(lf, dispGT, self.patchsize, refocus_flag)
            # if (scene_id == 4 or scene_id == 6 or scene_id == 15):
            #     glass_region = np.sum(dispGT_crop[:, :, 1]) < self.patchsize * self.patchsize
            if glass_region == False:
                sum_diff = np.sum(np.abs(lf_crop[self.angRes//2, self.angRes//2, :, :] -
                                         np.squeeze(lf_crop[self.angRes//2, self.angRes//2, self.patchsize//2, self.patchsize//2]))
                                  ) / (self.patchsize * self.patchsize)
        
        # lf_crop, dispGT_crop = random_crop(lf, dispGT, self.patchsize, refocus_flag)
        data = rearrange(lf_crop, 'a1 a2 h w -> (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        data, label = orientation_augmentation(data, dispGT_crop)
        data = data.astype('float32')
        label = label.astype('float32')
        data = ToTensor()(data.copy())
        label = ToTensor()(label.copy())
        # /////////////////////////////////////////////////////////////
        # 获取最大值和最小值
        # max_value1 = torch.max(data[~torch.isinf(data) & ~torch.isnan(data)])  # 忽略inf和NaN
        # min_value1 = torch.min(data[~torch.isinf(data) & ~torch.isnan(data)])  # 忽略inf和NaN
        # max_value2 = torch.max(label[~torch.isinf(label) & ~torch.isnan(label)])  # 忽略inf和NaN
        # min_value2 = torch.min(label[~torch.isinf(label) & ~torch.isnan(label)])  # 忽略inf和NaN
        # 检查inf和NaN
        # has_inf = torch.any(torch.isinf(data))
        # has_nan = torch.any(torch.isnan(data))

        # print("最大值:", max_value1.item())
        # print("最小值:", min_value1.item())
        # print("最大值:", max_value2.item())
        # print("最小值:", min_value2.item())
        # print("包含inf:", has_inf.item())
        # print("包含NaN:", has_nan.item())
        # /////////////////////////////////////////////////////////////
        return data, label

    def __len__(self):
        return self.item_num


def illuminance_augmentation(data):
    # 照明增强处理：包括色彩变换、亮度调整、加入随机噪声
    # 用于提高数据多样性与模型鲁棒性
    rand_3color = 0.05 + np.random.rand(3)
    # 随机生成三元组，值域[0.05, 1.05)
    # 代表RGB三色的混合权重，用于根据彩图生成灰度图
    rand_3color = rand_3color / np.sum(rand_3color)
    # 归一化
    R = rand_3color[0]
    G = rand_3color[1]
    B = rand_3color[2]
    data_gray = np.squeeze(R * data[:, :, :, :, 0] + G * data[:, :, :, :, 1] + B * data[:, :, :, :, 2])
    # 将RGB三色按随机权重加和，得到灰度图
    gray_rand = 0.4 * np.random.rand() + 0.8
    # 随机生成亮度，值域[0.8, 1.2)
    data_gray = pow(data_gray, gray_rand)
    # 通过幂运算随机调整亮度
    noise_rand = np.random.randint(0, 10)
    if noise_rand == 0:
        # 随机生成0到9的一个整数
        # 若为0（10%概率），则添加随机噪声
        gauss = np.random.normal(0.0, np.random.uniform() * np.sqrt(0.2), data_gray.shape)
        # 生成平均数为0，标准差随机，与灰度图shape相同的，符合正态分布的随机噪声
        data_gray = np.clip(data_gray + gauss, 0.0, 1.0)
        # 将灰度图的各像素值限制在0到1之间（超出边界的，直接用边界值替换）
    return data_gray


def refocus_augmentation(lf, dispGT):
    # 对光场图像与视差图进行随机重聚焦
    refocus_flag = 0
    # 用于指示是否执行了重聚焦操作
    refocus_rand = np.random.randint(0, 5)
    if refocus_rand == 0:
        # 随机生成0到4的随机整数
        # 若为0（20%），则进行重聚焦
        refocus_flag = 1
        angRes, _, h, w = lf.shape
        center = (angRes - 1) // 2
        # 中心视图编号
        min_d = int(np.min(dispGT[:, :, 0]))
        max_d = int(np.max(dispGT[:, :, 0]))
        # 根据实际视差图，求最大与最小视差值
        dispLen = 6 - (max_d - min_d)
        # dispLen为可在原视差基础上，进行增加或减少的视差值范围
        k = np.random.randint(dispLen + 1) - 3
        dd = k - min_d
        # dd为重聚焦造成的视差值变化量
        # 确保随机生成的dd为正为负的概率相近
        out_dispGT = np.zeros((h, w, 2), dtype=float)
        out_dispGT[:, :, 0] = dispGT[:, :, 0] + dd
        out_dispGT[:, :, 1] = dispGT[:, :, 1]
        # 将原视差图添加变化量作为输出
        # 不修改掩膜
        out_lf = np.zeros((angRes, angRes, h, w), dtype=float)
        for u in range(angRes):
            for v in range(angRes):
                dh, dw = dd * (u - center), dd * (v - center)
                # 求出不同视角下，根据视差变化量得出的像素移位量
                # 视差变化量与像素移位量成正比
                # 视差越大，即相机离物体越近，像素移位越明显
                if (dh > 0) & (dw > 0):
                    out_lf[u, v, 0:-dh-1, 0:-dw-1] = lf[u, v, dh:-1, dw:-1]
                elif (dh > 0) & (dw == 0):
                    out_lf[u, v, 0:-dh-1, :] = lf[u, v, dh:-1, :]
                elif (dh > 0) & (dw < 0):
                    out_lf[u, v, 0:-dh-1, -dw:-1] = lf[u, v, dh:-1, 0:dw-1]
                elif (dh == 0) & (dw > 0):
                    out_lf[u, v, :, 0:-dw-1] = lf[u, v, :, dw:-1]
                elif (dh == 0) & (dw == 0):
                    out_lf[u, v, :, :] = lf[u, v, :, :]
                elif (dh == 0) & (dw < 0):
                    out_lf[u, v, :, -dw:-1] = lf[u, v, :, 0:dw-1]
                elif (dh < 0) & (dw > 0):
                    out_lf[u, v, -dh:-1, 0:-dw-1] = lf[u, v, 0:dh-1, dw:-1]
                elif (dh < 0) & (dw == 0):
                    out_lf[u, v, -dh:-1, :] = lf[u, v, 0:dh-1, :]
                elif (dh < 0) & (dw < 0):
                    out_lf[u, v, -dh:-1, -dw:-1] = lf[u, v, 0:dh-1, 0:dw-1]
                else:
                    pass
    else:
        out_lf, out_dispGT = lf, dispGT

    return out_lf, out_dispGT, refocus_flag


def scale_augmentation(lf, dispGT, patchsize):
    if patchsize > 48:
        kk = np.random.randint(14)
    else:
        kk = np.random.randint(17)
    if (kk < 8):
        scale = 1
    elif (kk < 14):
        scale = 2 
        # 分辨率缩放至原来的1/2
    elif (kk < 17):
        scale = 3
        # 分辨率缩放至原来的1/3
    out_lf = lf[:, :, 0::scale, 0::scale]
    out_disp = dispGT[0::scale, 0::scale]
    # 每scale个像素采样一次
    # 下采样，会导致图像分辨率（尺寸）降低
    out_disp[:, :, 0] = out_disp[:, :, 0] / scale
    # 图像缩放后，各像素的原始视差值也要按相同比例缩放

    return out_lf, out_disp, scale


def orientation_augmentation(data, dispGT):
    if random.random() < 0.5:  # flip along W-V direction
        # 水平翻转
        data = data[:, ::-1]
        dispGT = dispGT[:, ::-1, :]
    if random.random() < 0.5:  # flip along H-U direction
        # 垂直翻转
        data = data[::-1, :]
        dispGT = dispGT[::-1, :, :]
    if random.random() < 0.5: # transpose between U-V and H-W
        # 旋转90度
        data = data.transpose(1, 0)
        dispGT = dispGT.transpose(1, 0, 2)

    return data, dispGT


def viewpoint_augmentation(data_in, angRes):
    if (angRes == 3):
        #u, v = np.random.randint(0, 7), np.random.randint(0, 7)
        u, v = 3, 3
    if (angRes == 5):
        #u, v = np.random.randint(0, 5), np.random.randint(0, 5)
        u, v = 2, 2
    if (angRes == 7):
        #u, v = np.random.randint(0, 3), np.random.randint(0, 3)
        u, v = 1, 1
    if (angRes == 9):
        u, v = 0, 0
    data_out = data_in[u : u + angRes, v : v + angRes, :, :]

    return data_out


def random_crop(lf, dispGT, patchsize, refocus_flag):
    # 对光场图像及其视差图进行随机裁剪,用于数据增强
    # patchsize: 裁剪后的图像尺寸（边长）
    angRes, angRes, h, w = lf.shape
    if refocus_flag == 1:
        bdr = 16
    else:
        bdr = 0
    h_idx = np.random.randint(bdr, h - patchsize - bdr)
    w_idx = np.random.randint(bdr, w - patchsize - bdr)
    out_lf = lf[:, :, h_idx : h_idx + patchsize, w_idx : w_idx + patchsize]
    out_disp = dispGT[h_idx : h_idx + patchsize, w_idx : w_idx + patchsize, :]

    return out_lf, out_disp



def ImageExtend(Im, bdr):
    # bdr: 上、下、左、右的所需拓展量
    # 通过图片拼接拓展原图尺寸
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])
    # 得到左右、上下、对角线翻转的图像
    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    # 拼接方式：
    # 对角 上下 对角
    # 左右 原图 左右
    # 对角 上下 对角
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]
    # 对拓展图像进行裁剪，使得尺寸=(h + bdr[0] + bdr[1], w + bdr[2] + bdr[3])
    return Im_out


def LFdivide(lf, patch_size, stride):
    '''操你妈看不懂'''
    # 按固定方式进行光场图像分割，用于数据预处理
    # patch_size: 分割所得的图像尺寸
    U, V, C, H, W = lf.shape
    data = rearrange(lf, 'u v c h w -> (u v) c h w')
    # 将视角维度合并，即 将各视角图像拼接为一个整体
    bdr = (patch_size - stride) // 2
    numU = (H + bdr * 2 - 1) // stride
    numV = (W + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    # 拓展图像尺寸至(H + patch_size - 1, W + patch_size - 1)
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(u v) (c h w) (n1 n2) -> n1 n2 u v c h w',
                      n1=numU, n2=numV, u=U, v=V, h=patch_size, w=patch_size)

    return subLF


def LFintegrate(subLFs, patch_size, stride):
    '''操你妈看不懂'''
    bdr = (patch_size - stride) // 2
    out = subLFs[:, :, :, bdr:bdr+stride, bdr:bdr+stride]
    out = rearrange(out, 'n1 n2 c h w -> (n1 h) (n2 w) c')

    return out.squeeze()