import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange   #rearrange用于重组张量



class Net(nn.Module):
    def __init__(self, angRes):
        # angRes：角分辨率，值为9
        super(Net, self).__init__()
        self.num_cascade = 2
        mindisp = -4    # 最小视差
        maxdisp = 4     # 最大视差
        self.angRes = angRes
        self.maxdisp = maxdisp
        self.init_feature = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            # 三维卷积：提取初步特征
            # Conv3d(输入图像通道1个, 
            #        输出图像通道16个（即总共使用16个卷积核分别进行16次卷积）, 
            #        卷积核1*3*3, 
            #        步长1, 
            #        三个维度分别填充0、1、1, 
            #        不往输出中添加可学习的偏差)
            nn.BatchNorm3d(16),
            # BatchNorm3d(输入通道16个)，输出通道数与输入一致
            ResB(16), ResB(16), ResB(16), ResB(16),
            ResB(16), ResB(16), ResB(16), ResB(16),
            # 八个残差块堆叠：提取深层次特征
            nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # LeakyReLU(x为负时的斜率, 
            #           允许原地修改张量)
            nn.Conv3d(16, 8, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.lastconv = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.build_cost = BuildCost(8, 512, angRes, mindisp, maxdisp)
        self.aggregate = Aggregate(512, 160, mindisp, maxdisp)

    def forward(self, x, dispGT=None):
        # dispGT: ground truth disparity map
        # x: 传入的batch_size个光场数据
        lf = rearrange(x, 'b c (a1 h) (a2 w) -> b c a1 a2 h w', a1=self.angRes, a2=self.angRes)
        # 用字符串代表张量的原始结构，以及其重组后的结构。括号表示压缩至同一维度的多个维度
        # a1=a2=9
        x = rearrange(x, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        # 重塑x形状，使之符合conv3d的预期输入shape
        b, c, _, h, w = x.shape
        feat = self.init_feature(x)
        feat = self.lastconv(feat)
        if dispGT is not None:
            mask = Generate_mask(lf, dispGT)
            # 有准确视差图时，可直接进行掩膜生成
            cost = self.build_cost(feat, mask)
            disp = self.aggregate(cost)
        else:
            mask = torch.ones(1, self.angRes ** 2, h, w).to(x.device)
            cost = self.build_cost(feat, mask)
            disp = self.aggregate(cost)
            # 无准确视差图时，采用初始全1掩膜以计算初始视差图
            mask = Generate_mask(lf, disp)
            cost = self.build_cost(feat, mask)
            disp = self.aggregate(cost)

        return disp


class BuildCost(nn.Module):
    def __init__(self, channel_in, channel_out, angRes, mindisp, maxdisp):
        super(BuildCost, self).__init__()
        self.oacc = ModulateConv2d(channel_in, channel_out, kernel_size=angRes, stride=1, bias=False)
        self.angRes = angRes
        self.mindisp = mindisp          # 值为-4
        self.maxdisp = maxdisp          # 值为4
        self.channel_att = channel_out  # 值为512
        self.channel_in = channel_in    # 值为8

    def forward(self, x, mask):
        # 输入：特征图x, 掩膜mask
        b, c, aa, h, w = x.shape
        x = rearrange(x, 'b c (a1 a2) h w -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        bdr = (self.angRes // 2) * self.maxdisp
        # bdr: 填充量，值为16
        pad = nn.ZeroPad2d((bdr, bdr, bdr, bdr))
        # ZeroPad2d((left,      左填充像素数
        #            right,     右填充像素数
        #            top,       上填充像素数
        #            bottom))   下填充像素数
        # return: 所设定的零填充方法，对张量的最后两维进行零填充
        # 例：[b, c, h, w] -> [b, c, h + top + bottom, w + left + right]
        x_pad = pad(x)
        x_pad = rearrange(x_pad, '(b a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        # 将所有视角下的图像拼接成一整个图像
        h_pad, w_pad = h + 2 * bdr, w + 2 * bdr     # 值为512 + 2 * 16
        mask_avg = torch.mean(mask, dim=1)
        # 按通道维度求平均值，即将81个视角的掩膜平均化为一个
        cost = []
        for d in range(self.mindisp, self.maxdisp + 1):
            dila = [h_pad - d, w_pad - d]
            # dila：扩张率
            self.oacc.dilation = dila
            # 将对应视差下的扩张率传入OACC，用于在后续成本构建中进行angular patch
            crop = (self.angRes // 2) * (d - self.mindisp)
            # crop：裁剪量，确保匹配成本的计算仅基于有匹配关系的特征区域
            # 视差越大，边缘视角中无法匹配的区域越多，所需的裁剪量越大
            if d == self.mindisp:
                feat = x_pad
            else:
                feat = x_pad[:, :, crop: -crop, crop: -crop]
                # 裁剪掉特征图中无法匹配的边缘区域
            current_cost = self.oacc(feat, mask)
            # 计算 (ω<k> * A<p,d>(k) * ∆m<p,k>) 见匹配成本公式
            cost.append(current_cost / mask_avg.unsqueeze(1).repeat(1, current_cost.shape[1], 1, 1))
            # 计算当前视差下，所有视图的匹配成本
        cost = torch.stack(cost, dim=2)
        # 将各视差下的匹配成本拼接

        return cost


class Aggregate(nn.Module):
    def __init__(self, inC, channel, mindisp, maxdisp):
        # inC: 输入通道数，512
        # channel: 中间通道数，160
        super(Aggregate, self).__init__()
        self.sq = nn.Sequential(
            nn.Conv3d(inC, channel, 1, 1, 0, bias=False), 
            nn.BatchNorm3d(channel), 
            nn.LeakyReLU(0.1, inplace=True))
        # kernel_size = 1, stride = 1, padding = 0
        # kernel_size取整数，表示各维度尺寸相同
        # 使用1*1*1卷积核：在不改变空间维度的前提下，减少通道数，并实现跨通道的信息交互
        self.Conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), 
            nn.BatchNorm3d(channel), 
            nn.LeakyReLU(0.1, inplace=True))
        self.Conv2 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), 
            nn.BatchNorm3d(channel), 
            nn.LeakyReLU(0.1, inplace=True))
        self.Resb1 = ResB3D(channel)
        self.Resb2 = ResB3D(channel)
        # 两个三维残差块
        self.Conv3 = nn.Sequential(
            nn.Conv3d(channel, channel, 3, 1, 1, bias=False), 
            nn.BatchNorm3d(channel), 
            nn.LeakyReLU(0.1, inplace=True))
        self.Conv4 = nn.Conv3d(channel, 1, 3, 1, 1, bias=False)
        # 减少通道数至1，以得到最终的匹配成本
        self.softmax = nn.Softmax(1)
        # 将匹配成本转化为概率分布（注意力图）
        self.mindisp = mindisp
        self.maxdisp = maxdisp

    def forward(self, psv):
        buffer = self.sq(psv)
        buffer = self.Conv1(buffer)
        buffer = self.Conv2(buffer)
        buffer = self.Resb1(buffer)
        buffer = self.Resb2(buffer)
        buffer = self.Conv3(buffer)
        score = self.Conv4(buffer)
        attmap = self.softmax(score.squeeze(1))
        temp = torch.zeros(attmap.shape).to(attmap.device)
        for d in range(self.maxdisp - self.mindisp + 1):
            # d in [0, 9)
            temp[:, d, :, :] = attmap[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)
        # (在attmap未经过self.Conv4将通道数压缩至1的前提下)
        # 将attmap中的元素，视为不同视差、不同像素下的权重
        # 即相当于求出每个像素的加权平均视差
        return disp


class ResB(nn.Module):
    def __init__(self, feaC):
        super(ResB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(feaC, feaC, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(feaC, feaC, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(feaC))

    def forward(self, x):
        out = self.conv(x)
        return x + out
        # return: 当前残差块的输入（即上一残差块的输出）+ 当前残差块的输出
        # 避免了网络堆叠层数过多引起的梯度消失或爆炸问题


class ResB3D(nn.Module):
    # 引入了通道注意力机制的残差块
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels))
        self.calayer = CALayer(channels, 9)

    def forward(self, x):
        buffer = self.body(x)
        return self.calayer(buffer) + x
        # 将经过通道加权后的本层输入，与本层输出相加作为输出

class CALayer(nn.Module):
    # 通道注意力层
    def __init__(self, channel, num_views):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((num_views, 1, 1))
        # 自适应平均池化层
        # 将输入维度由 [b, c, h, w] 池化为 [b, num_views, 1, 1]
        # 保留通道信息的同时，减少每个通道内空间维度上的信息，从而得到各通道的描述符
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // 16, 1, 1, 0, bias=True),
                nn.BatchNorm3d(channel // 16),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(channel // 16, channel, 1, 1, 0, bias=True),
                nn.BatchNorm3d(channel),
                nn.Sigmoid())
        # 对通道描述符进行特征提取，得到各通道的重要性描述（通道权重）

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
        # 将通道权重作用于输入x，增强对重要特征的响应

def Generate_mask(lf, disp):
    b, c, angRes, _, h, w = lf.shape
    x_base = torch.linspace(0, 1, w).repeat(1, h, 1).to(lf.device)
    # 在[0, 1]之间等间距生成 w 个数，组成一维张量
    # 将该张量重复 1*h*1 次，形成新张量
    # 实际意义：视图中每个像素经过归一化后的横坐标
    # x_base.shape = [1, h, w]
    y_base = torch.linspace(0, 1, h).repeat(1, w, 1).transpose(1, 2).to(lf.device)
    # transpose(1, 2): 将维度1、维度2互换（编号从0开始）
    # 实际意义：视图中每个像素经过归一化后的纵坐标
    # y_base.shape = [1, h, w]
    center = (angRes - 1) // 2 # center = 4
    img_ref = lf[:, :, center, center, :, :]
    # 实际的中心视图
    img_res = [] 
    # 残差图
    for u in range(angRes):
        for v in range(angRes):
            # 遍历所有视图
            img = lf[:, :, u, v, :, :]
            # img.shape = [b, c, h, w]
            if (u == center) & (v == center):
                img_warped = img
                # 如果该视图即为中心视图，则不进行矫正操作
            else:
                du, dv = u - center, v - center
                img_warped = warp(img, -disp, du, dv, x_base, y_base)
            img_res.append(abs((img_warped - img_ref)))
            # 收集每个视图与实际中心视图的残差图
    mask = torch.cat(img_res, dim=1)
    # 按channel维度，将所有残差图拼接为掩膜
    # mask.shape = [b, u*v, h, w]
    out = (1 - mask) ** 2
    return out


def warp(img, disp, du, dv, x_base, y_base):
    # 将非中心视图矫正为理论上的中心视图
    b, _, h, w = img.size()
    x_shifts = dv * disp[:, 0, :, :] / w
    y_shifts = du * disp[:, 0, :, :] / h
    # 实际意义：该视图中每个像素相对于中心视图的横纵坐标偏移量（经过了归一化）
    # 计算方法：该视图与中心视图在SAI阵列中的坐标差，与视差图中每个像素的视差值相乘，最后归一化
    # 操作 "/ w" 与 "/ h" 即为归一化操作
    flow_field = torch.stack((x_base + x_shifts, y_base + y_shifts), dim=3)
    # flow_field: 该视图中每个像素矫正至中心视图后的归一化坐标（原坐标 + 偏移量）
    # flow_field.shape = [1, h, w, 2], 即 h*w 个 坐标(x, y)
    img_warped = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')
    # img_warped: 理论上的中心视图
    # grid_sample: 网格采样操作
    # grid_sample会对img建立坐标系（采用归一化坐标），根据flow_field提供的坐标对img进行采样
    # 操作 "2 * flow_field - 1" 用于将flow_field的值域由[0, 1]转换为[-1,1]，以符合网格采样坐标系要求
    return img_warped


class ModulateConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, dilation=1, bias=False):
        super(ModulateConv2d, self).__init__()
        self.kernel_size = kernel_size  # [9, 9]
        self.stride = stride
        self.dilation = dilation
        self.flatten = nn.Unfold(kernel_size=1, stride=1, dilation=1, padding=0)
        # nn.Unfold(卷积核大小=1, 步长=1, 膨胀率=1, 无填充)
        # 将卷积核所能覆盖到的每一片区域（可重叠）展开为一维向量，并将这些一维向量组合成新的二维张量
        # [b, c, h, w] -> [b, c*k*k, l] 其中k*k为卷积核大小，l为卷积核所能覆盖到的区域总数
        self.fuse = nn.Conv2d(channel_in * kernel_size * kernel_size, channel_out,
                              kernel_size=1, stride=1, padding=0, bias=bias, groups=channel_in)

    def forward(self, x, mask):
        # x: 所有视角拼接成的整体图像的特征图（已裁掉无法匹配的边缘部分）
        # mask.shape = [b, u*v, h, w]
        mask_flatten = self.flatten(mask)
        # 将掩膜展平：[b, u*v, h, w] -> [b, u*v, h*w]
        Unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        x_unfold = Unfold(x)
        # 将整体特征图按照给定扩张率构建angular patch
        # x_unfold为给定扩张率下可构建的所有angular patch的集合
        x_unfold_modulated = x_unfold * mask_flatten.repeat(1, x.shape[1], 1)
        # 用掩膜对所有angular_patch进行调制
        Fold = nn.Fold(output_size=(mask.shape[2], mask.shape[3]), kernel_size=1, stride=1)
        x_modulated = Fold(x_unfold_modulated)
        # 将经调制的angular_patch复原，生成经调制的特征图
        # 至此，特征图中各视角的对应像素，都经过了mask中相对应的权重的调制
        out = self.fuse(x_modulated)
        # 对经调制的特征图进行卷积，得到当前扩张率下的匹配成本
        # out的本质是经过了mask调制后的特征图
        return out


if __name__ == "__main__":
    angRes = 9
    net = Net(angRes).cuda()
    from thop import profile
    # 用于评估模型的参数数量与浮点运算（FLOPs）数量
    input = torch.randn(1, 1, 32 * angRes, 32 * angRes).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))