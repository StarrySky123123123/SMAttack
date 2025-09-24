import os

import torch.nn
from torch.utils.data import DataLoader

from tools.loadModel import *
from tools.ThinPlateSpline import *
from feeder.feeder import *
from feeder.ctrgcn.ctrgcn_feeder_ntu import Feeder as ctrgcn_feeder

import argparse
from omegaconf import OmegaConf
import time

import numpy as np



def compute_skeleton_pos_cov(data):
    N, C, T, V, M = data.shape
    data = data[..., 0]
    cov_list = []
    for i in range(N):
        pos = data[i]
        pos = pos.permute(1, 2, 0)
        pos = pos.reshape(pos.shape[0], -1)
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        cov = (pos_centered.t() @ pos_centered) / (pos_centered.shape[0] - 1)
        cov_list.append(cov)

    cov_array = torch.stack(cov_list, dim=0)
    return cov_array


def foolRateCal(args, rlabels, flabels, logits=None):
    hitIndices = []

    if args.attackType == 'ab':
        for i in range(0, len(flabels)):
            if flabels[i] != rlabels[i]:
                hitIndices.append(i)
    elif args.attackType == 'abn':
        for i in range(len(flabels)):
            sorted, indices = torch.sort(logits[i], descending=True)
            ret = (indices[:topN] == rlabels[i]).nonzero(as_tuple=True)[0]
            if len(ret) == 0:
                hitIndices.append(i)
    elif args.attackType == 'sa':
        for i in range(0, len(flabels)):
            if flabels[i] == rlabels[i]:
                hitIndices.append(i)

    return len(hitIndices) / len(flabels) * 100


def getRA(data_freq, Len=300):
    # 计算单边谱
    F_single_sided = (2 / Len) * data_freq  # 只取正频部分，并加权
    # 获取幅度和相位
    R = torch.abs(F_single_sided)  # 幅度
    A = torch.angle(F_single_sided)  # 相位
    return R, A


def complex_freq(R, A, Len=300):
    F_prime = R * torch.exp(1j * A)
    F_prime = F_prime * Len / 2
    return F_prime


def project_space_time(data_freq):
    OriginData = torch.fft.ifft(data_freq, dim=2).real
    return OriginData

topN = 3
global Vs

def get_sample_points(data):
    global Vs
    N, C, T, V, M = data.shape
    points = torch.zeros([N, C, T, len(Vs), M])
    for i in range(len(Vs)):
        points[:, :, :, i, :] = data[:, :, :, Vs[i], :]
    points = points.permute(0, 2, 4, 3, 1).contiguous().view(N * T * M, len(Vs), C)
    return points

def compute_Pi(input_tensor):
    squared_magnitudes = input_tensor.norm(p=2, dim=(1, 2, 3, 4)) ** 2
    P_i = torch.sqrt(squared_magnitudes / input_tensor.size(0))
    return P_i


def unspecificAttack(labels, classNum):
    flabels = np.ones((len(labels), classNum))
    flabels = flabels * 1 / classNum
    return torch.LongTensor(flabels)


def specifiedAttack(labels, classNum, targettedClasses=[]):
    if len(targettedClasses) <= 0:
        flabels = torch.LongTensor(np.random.randint(0, classNum, len(labels)))
    else:
        flabels = targettedClasses
    return flabels


def loadData(model_name, data_path, label_path, num_frame_path, batch_size, args, label_type=1):
    if (model_name == 'ctrgcn' or model_name == 'ctrgcn120' or model_name == 'hdgcn' or model_name == 'hdgcn120'):
        frame_len = 64
        feeder = ctrgcn_feeder(data_path, p_interval=[0.95], window_size=64, split='train')
    elif (model_name == 'ctrgcn_kinetics' or model_name == 'selfgcn_kinetics' or model_name == 'hdgcn_kinetics'):
        frame_len = 64
        feeder = Feeder(data_path, label_path, label_type=label_type, window_size=64, random_choose=True, random_move=False)
    else:
        frame_len = 300
        feeder = Feeder(data_path, label_path, num_frame_path=num_frame_path, label_type=label_type)
    trainloader = DataLoader(feeder,
                             batch_size=batch_size,
                             num_workers=4, pin_memory=True, drop_last=True, shuffle=False)

    return trainloader, frame_len


def attack(args):
    global Vs
    if args.dataset == 'ntu60' or args.dataset == 'ntu120':
        spine_points = [0, 1, 20]
        Vs = [0, 1, 2, 3,
              7, 11, 15, 19,
              9, 13, 17, 21]
    else:
        # kinetics
        spine_points = [0]
        Vs = [0, 1, 3, 5,
              4, 7, 8, 11,
              10, 13]
    if not os.path.exists(f'{args.save_path}'):
        os.makedirs(f'{args.save_path}')

    trainloader, frame_len = loadData(args.model_name, args.data_path, args.label_path, args.num_frame_path, args.batch_size, args, 1)

    epochs = args.epochs
    deltaT = 1 / 30

    model = getModel(args.model_name)

    SourceSample = []
    AttackSample = []
    FramesSample = []
    LabelSample = []

    overallFoolRate = 0
    batchTotalNum = 0
    total_time = 0

    # 查找有效数据
    if args.dataset == 'kinetics':
        if not os.path.exists(f'./data/kinetics/{args.model_name.split("_")[0]}/'):
            os.makedirs(f'./data/kinetics/{args.model_name.split("_")[0]}/')
        if not os.path.exists(f'./data/kinetics/{args.model_name.split("_")[0]}/datas.npy'):
            model.eval()
            datas = []
            frames = []
            labels = []
            for batch_idx, (data, target, frame) in enumerate(trainloader):
                data = data.cuda()
                target = target.cuda()
                pred = torch.argmax(model(data), dim=1)
                for i in range(len(target)):
                    if pred[i] == target[i]:
                        datas.append(data[i].detach().clone().cpu())
                        frames.append(frame[i].detach().clone().cpu())
                        labels.append(target[i].detach().clone().cpu())

            np.save(f'./data/kinetics/{args.model_name.split("_")[0]}/datas.npy', np.stack(datas))
            np.save(f'./data/kinetics/{args.model_name.split("_")[0]}/frames.npy', np.stack(frames))
            np.save(f'./data/kinetics/{args.model_name.split("_")[0]}/labels.npy', np.stack(labels))
            print(f'数据预处理完毕,总共有{len(datas)}条数据')
        trainloader, frame_len = loadData(args.model_name, f'./data/kinetics/{args.model_name.split("_")[0]}/datas.npy',
                                             f'./data/kinetics/{args.model_name.split("_")[0]}/labels.npy',
                                             f'./data/kinetics/{args.model_name.split("_")[0]}/frames.npy',
                                             args.batch_size, args, label_type=-1)

    for batchNo, (data, label, frame) in enumerate(trainloader):
        data = data.cuda()
        label = label.cuda()

        if args.attackType == 'abn':
            flabels = unspecificAttack(label, args.classNum).cuda()
        elif args.attackType == 'sa':
            flabels = specifiedAttack(label, args.classNum).cuda()
        elif args.attackType == 'ab':
            flabels = label
        else:
            print('specified targetted attack, no implemented')
            return

        N, C, T, V, M = data.shape
        ori_cov = compute_skeleton_pos_cov(data)
        adData = data.detach().clone()
        trajectory_freq = torch.fft.fft(adData, dim=2)
        R, A = getRA(trajectory_freq, frame_len)
        R_max = torch.max(R)
        # 归一化 R[ω]
        S_r = (R / R_max)

        # 初始化为0
        weights = torch.zeros(size=R.shape).cuda()
        adWeights = weights.clone()
        adWeights.requires_grad = True

        source_space_points = get_sample_points(adData)

        learning_rate = args.lr
        start_time = time.time()  # 记录开始时间
        batchTotalNum += 1
        for epoch in range(epochs):
            with torch.no_grad():
                # 对抗部分，每次都计算出一个新的样本
                R_p = R + adWeights
                R_p = torch.clamp(R_p, 0, torch.max(R_p).item())
                proData = project_space_time(complex_freq(R_p, A, frame_len))
                # 对脊椎进行终极约束
                proData.data[:, :, :, spine_points, :] = torch.clamp(proData.data[:, :, :, spine_points, :]
                                                                     ,data.data[:, :, :, spine_points, :]-0.001
                                                                     ,data.data[:, :, :, spine_points, :]+0.001)
                target_points = get_sample_points(proData)
                # PSM只负责拟合不负责对抗，负责拟合新的样本
                tps = ThinPlateSpline(source_space_points, target_points) # TPS负责空间平滑
                tpData = adData.permute(0, 2, 4, 3, 1).contiguous().view(N * T * M, V, C)
                tpData = tps.transform(tpData)
                tpData = tpData.view(N, T, M, V, C).permute(0, 4, 1, 3, 2)

            # 重新组织数据来计算weights的梯度
            freq_tp = torch.fft.fft(tpData, dim=2)
            R_tp, A_tp = getRA(freq_tp, frame_len)
            adWeights = R_tp - R
            adWeights.requires_grad = True
            R_p = R + adWeights
            R_p = torch.clamp(R_p, 0, torch.max(R_p).item())
            newData = project_space_time(complex_freq(R_p, A, frame_len)).cuda()
            pred = model(newData)
            pred = pred / torch.max(pred, dim=1, keepdim=True)[0]
            predictedLabels = torch.argmax(pred, dim=1)

            if args.attackType == 'ab':
                foolRate = foolRateCal(args, flabels, predictedLabels)
            elif args.attackType == 'abn':
                foolRate = foolRateCal(args, label, predictedLabels, pred)
            elif args.attackType == 'sa':
                cFlabels = label
                foolRate = foolRateCal(args, cFlabels, predictedLabels)
            else:
                print('specified targetted attack, no implemented')
                return

            if args.task == 'transfer':
                if epoch == epochs - 1:
                    for i in range(len(label)):
                        if label[i] != predictedLabels[i]:
                            SourceSample.append(data[i].detach().clone().cpu())
                            AttackSample.append(newData[i].detach().clone().cpu())
                            FramesSample.append(frame[i].detach().clone().cpu())
                            LabelSample.append(label[i].detach().clone().cpu())
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    total_time += elapsed_time
                    print(
                        f'sample number:{len(SourceSample)} batchNo:{batchNo} average time:{total_time / (batchNo + 1)}')
                    break

            if args.task != 'transfer':
                if foolRate == 100 or epoch == epochs - 1:
                    for i in range(len(label)):
                        if label[i] != predictedLabels[i]:
                            SourceSample.append(data[i].detach().clone().cpu())
                            AttackSample.append(newData[i].detach().clone().cpu())
                            FramesSample.append(frame[i].detach().clone().cpu())
                            LabelSample.append(label[i].detach().clone().cpu())
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    total_time += elapsed_time
                    print(
                        f'sample number:{len(SourceSample)} batchNo:{batchNo} average time:{total_time / (batchNo + 1)}')
                    break

            adv_acc = (newData[:, :, 2:, Vs, :] - 2 * newData[:, :, 1:-1, Vs, :] + newData[:, :, :-2, Vs,
                                                                                      :]) / deltaT / deltaT

            smooth_loss = torch.sqrt(torch.mean((adv_acc) ** 2))

            adv_cov = compute_skeleton_pos_cov(newData)
            cov_loss = torch.sum((adv_cov - ori_cov)**2)

            if args.attackType == 'ab':
                classLoss = -torch.nn.CrossEntropyLoss()(pred, flabels)
            elif args.attackType == 'abn':
                classLoss = torch.mean((pred - flabels) ** 2)
            else:
                classLoss = torch.nn.CrossEntropyLoss()(pred, flabels)
            loss = 0.6*classLoss + 0.1*cov_loss + 0.3*smooth_loss

            adWeights.grad = None
            loss.backward()
            cgs = adWeights.grad

            cgsView = cgs.contiguous().view(cgs.shape[0], -1)
            cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
            cgsView /= cgsnorms[:, np.newaxis]

            with torch.no_grad():
                missedIndices = []
                if args.attackType == 'ab':
                    if args.task == 'transfer':
                        for i in range(len(label)):
                            missedIndices.append(i)
                    else:
                        for i in range(len(label)):
                            if label[i] == predictedLabels[i]:
                                missedIndices.append(i)
                elif args.attackType == 'abn':
                    for i in range(len(label)):
                        sorted, indices = torch.sort(pred[i], descending=True)
                        ret = (indices[:topN] == label[i]).nonzero(as_tuple=True)[0]
                        if len(ret) > 0:
                            missedIndices.append(i)
                elif args.attackType == 'sa':
                    for i in range(len(label)):
                        if label[i] != predictedLabels[i]:
                            missedIndices.append(i)
                adWeights[missedIndices] = adWeights[missedIndices] - cgs[missedIndices].sign() * learning_rate

            if epoch % 20 == 0:
                print(
                    f'epoch:{epoch} loss={loss} cov_loss={cov_loss} smooth_loss={smooth_loss} foolRate={foolRate}')

        overallFoolRate += foolRate
        print(f"Current fool rate is {overallFoolRate / batchTotalNum}")

    # 保存
    if len(SourceSample) != 0:
        np.save(f'{args.save_path}/SourceSample.npy', np.stack(SourceSample))
        np.save(f'{args.save_path}/AttackSample.npy', np.stack(AttackSample))
        np.save(f'{args.save_path}/FramesSample.npy', np.stack(FramesSample))
        np.save(f'{args.save_path}/LabelSample.npy', np.stack(LabelSample))
        print(f'save sample, sample sum number:{len(SourceSample)}')


if __name__ == "__main__":
    print('SMAttack')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--attackType", type=str, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.defend = args.defend
    config.aux_model_name = args.aux_model_name
    if args.attackType != None:
        config.attackType = args.attackType
    print(f"\033[31m{config}\033[0m")
    attack(config)
