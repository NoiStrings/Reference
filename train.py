import time
import os
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from tqdm import tqdm
from model import Net


# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd',           type=str,   default=os.getcwd())
    parser.add_argument('--log_path',      type=str,   default=os.getcwd() + "/logs/")
    parser.add_argument('--device',        type=str,   default='cuda:0')
    parser.add_argument('--parallel',      type=bool,  default=False)
    parser.add_argument('--num_workers',   type=int,   default=6)
    parser.add_argument("--angRes",        type=int,   default=9,    help="angular resolution")
    parser.add_argument('--model_name',    type=str,   default='OACC-Net')
    parser.add_argument('--trainset_dir',  type=str,   default='./data/training/')
    parser.add_argument('--validset_dir',  type=str,   default='./data/validation/')
    parser.add_argument('--patchsize',     type=int,   default=48)
    parser.add_argument('--batch_size',    type=int,   default=8)
    parser.add_argument('--lr',            type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--n_epochs',      type=int,   default=100, help='number of epochs to train')
    # 最多训练多少个epoch
    parser.add_argument('--n_steps',       type=int,   default=3500, help='number of epochs to update learning rate')
    # 每多少个epoch更新一次学习率
    parser.add_argument('--gamma',         type=float, default=0.5,  help='learning rate decaying factor')
    # 学习率衰减率，用于学习率调度器
    parser.add_argument('--load_pretrain', type=bool,  default=True)
    parser.add_argument('--model_path',    type=str,   default='./models/OACC-Net.pth.tar')

    return parser.parse_args()

def train(cfg):
    if cfg.parallel:
        cfg.device = 'cuda:0'
        # 若采用并行计算，则使用第一个GPU
    net = Net(cfg.angRes)
    net.to(cfg.device)
    # 初始化模型，并将其移动至指定设备上
    cudnn.benchmark = True
    # 启用cudnn性能优化
    epoch_state = 0

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            # map_location: 将预训练模型中所有在 CUDA 设备 0 上的张量加载到 CPU 上
            net.load_state_dict(model['state_dict'], strict=False)
            # 将预训练模型的状态字典（模型参数）载入net
            # strict=False: 允许net与预训练模型的权重存在一定差异
            epoch_state = model['epoch']
        else:
            print("=> no model found at '{}'".format(cfg.model_path))

    if cfg.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1])
        # 允许模型在0、1两块GPU上运行
    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    # 定义损失函数，并移至设备上
    optimizer = torch.optim.Adam([paras for paras in net.parameters() 
                                  if paras.requires_grad == True], lr=cfg.lr)
    # 定义优化器，只优化需要优化的参数
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=cfg.n_steps, gamma=cfg.gamma)
    # 创建StepLR学习率调度器，用于按给定的周期与比率减少学习率
    # step_size: 每隔多少epoch减少一次学习率
    # gamma: 衰减率
    scheduler._step_count = epoch_state
    # 设置学习率调度器的起始epoch
    loss_list = []
    # 损失列表，记录每个epoch的损失

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        # 每轮循环为一个epoch
        train_set = TrainSetLoader(cfg)
        train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, 
                                  batch_size=cfg.batch_size, shuffle=True)
        # num_workers: 加载数据时使用的进程数
        # shuffle: 每个epoch开始时打乱数据
        loss_epoch = []
        # 记录当前epoch的各batch的损失
        for idx_iter, (data, dispGT) in tqdm(enumerate(train_loader), 
                                             total=len(train_loader)):
            # train_loader每执行一次，生成batch_size个光场图像及其对应的视差图
            # tqdm用于生成可视化进度条(每次进度条完成即为一个epoch结束)
            data, dispGT = data.to(cfg.device), dispGT.to(cfg.device)
            disp = net(data, dispGT)
            # 将当前batch的训练数据移至指定设备，并输入模型
            # 输出预测的视差图
            loss = criterion_Loss(disp, dispGT[:, 0, :, :].unsqueeze(1))
            # 预测视差图与实际视差图进行比对，计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 清除上次优化的梯度、执行反向传播计算新梯度、更新模型参数
            loss_epoch.append(loss.data.cpu())
            # 收集当前batch的损失

        if idx_epoch % 1 == 0:
            # 每个epoch执行一次
            loss_list.append(float(np.array(loss_epoch).mean()))
            log_info = "[Train] " + time.ctime()[4:-5] + "\t epoch: {:0>4} | loss: {:.5f}".format(idx_epoch + 1, float(np.array(loss_epoch).mean()))
            with open("logs/train_log.txt", "a") as f:
                f.write(log_info)
                f.write("\n")
            print(log_info)  
            '''print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % 
                  (idx_epoch + 1, float(np.array(loss_epoch).mean())))'''
            # 计算、收集、打印当前epoch各batch的损失平均值
            if cfg.parallel:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='./models/', filename=cfg.model_name + '.pth.tar')
            else:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                }, save_path='./models/', filename=cfg.model_name + '.pth.tar')
            # 保存模型状态（当前周期索引与状态字典）
            # 并行训练需通过net.module.state_dict()获取
        if idx_epoch % 10 == 9:
            # 每10个epoch执行一次
            if cfg.parallel:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='./models/', filename=cfg.model_name + str(idx_epoch + 1) + '.pth.tar')
            else:
                save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
            }, save_path='./models/', filename=cfg.model_name + str(idx_epoch + 1) + '.pth.tar')
            
            valid(net, cfg, idx_epoch + 1)
            # 进行模型验证

        scheduler.step()
        # 更新优化器的学习率

def valid(net, cfg, epoch):

    torch.no_grad()
    scene_list = ['boxes', 'cotton', 'dino', 'sideboard']
    angRes = cfg.angRes
    
    with open("logs/valid_log.txt", "a") as f:
        f.write("epoch: {:0>4} ==========".format(epoch))
        f.write("\n")
    
    for scenes in scene_list:
        # 循环4次，每次为一个场景
        lf = np.zeros(shape=(9, 9, 512, 512, 3), dtype=int)
        for i in range(81):
            temp = imageio.imread(cfg.validset_dir + scenes + '/input_Cam0%.2d.png' % i)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
            del temp
            # 读取当前场景的所有视角
        lf = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
        # 对各像素值进行归一化，并通过取平均值进行通道合并
        disp_gt = np.float32(
            read_pfm(cfg.validset_dir + scenes + '/gt_disp_lowres.pfm'))  # load groundtruth disparity map
        # 读取实际视差图
        
        angBegin = (9 - angRes) // 2
        lf_angCrop = lf[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

        patchsize = 128
        stride = patchsize // 2
        mini_batch = 4

        data = torch.from_numpy(lf_angCrop)
        sub_lfs = LFdivide(data.unsqueeze(2), patchsize, stride)
        # 将光场图分割为若干子图
        n1, n2, u, v, c, h, w = sub_lfs.shape
        sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
        num_inference = (n1 * n2) // mini_batch
        # 计算mini_batch数目（每个mini_batch包含4张子图）
        with torch.no_grad():
            out_disp = []
            for idx_inference in range(num_inference):
                # 每轮循环为一个mini_batch
                current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :]
                # 读取当前mini_batch包含的子图
                input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                out_disp.append(net(input_data.to(cfg.device)))
                # 输入当前mini_batch至模型，收集预测的深度图

            if (n1 * n2) % mini_batch:
                # 若存在不足以构成mini_batch的剩余的子图，操作同上
                current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                out_disp.append(net(input_data.to(cfg.device)))

        out_disps = torch.cat(out_disp, dim=0)
        out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
        disp = LFintegrate(out_disps, patchsize, patchsize // 2)
        # 合并之前分割的子图
        disp = disp[0: data.shape[2], 0: data.shape[3]]
        # 裁剪合并后的图，以确保其shape与标准一致
        disp = np.float32(disp.data.cpu())

        mse100 = np.mean((disp[11:-11, 11:-11] - disp_gt[11:-11, 11:-11]) ** 2) * 100
        # 选取靠近中心的像素，计算均方根误差
        # 边缘像素可能因为缺少邻域信息而导致预测失真
        # txtfile = open(cfg.model_name + '_MSE100.txt', 'a')
        # txtfile.write('mse_{}={:3f}\t'.format(scenes, mse100))
        # txtfile.close()
        
        log_info = "[Valid] " + time.ctime()[4:-5] + "\t scene: {:<11} | loss: {:.5f}".format(scenes, mse100)
        with open("logs/valid_log.txt", "a") as f:
            f.write(log_info)
            f.write("\n")
        print(log_info)

    return


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))


if __name__ == '__main__':
    cfg = parse_args()
    try:
        os.makedirs("log")
    except:
        pass
    train(cfg)
