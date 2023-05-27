import os
import argparse
from utils import *
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_rednet import REDNet20
from model_dncnn import DnCNN
from dataset import prepare_data, Dataset

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为0号设备

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Denoise")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')  # 第一次运行时设置为True
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--out", type=str, default="logs/DnCNN", help='path of log files')  # 参数存储路径，自己设置
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--model", type=str, default='DnCNN', help='type of the model')  # 换模型调整此项
opt = parser.parse_args()


def main():
    print('Loading dataset ...\n')  # Load dataset
    dataset_train = Dataset(train=True)  # training dataset
    dataset_val = Dataset(train=False)  # validate dataset
    loader_train = DataLoader(dataset=dataset_train, num_workers=2, batch_size=opt.batchSize, shuffle=True)
    # num_workers的经验设置值是自己电脑/服务器的CPU核心数
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    if opt.model == 'REDNet':
        net = REDNet20()
    elif opt.model == 'DnCNN':
        net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
        net.apply(weights_init_kaiming)  # 为DnCNN设置的初始化权重

    criterion = nn.MSELoss(reduction='sum')  # 损失函数

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)  # Move to GPU
    criterion.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    for epoch in range(opt.epochs):  # set learning rate
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)  # 噪声
            noisy_image = img_train + noise
            img_train, noisy_image = Variable(img_train.to(device)), Variable(noisy_image.to(device))
            out_train = model(noisy_image)  # output
            if opt.model == 'REDNet':
                loss = criterion(out_train, img_train) / (noisy_image.size()[0] * 2)
            elif opt.model == 'DnCNN':
                noise = Variable(noise.to(device))
                loss = criterion(out_train, noise) / (noisy_image.size()[0] * 2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            if opt.model == 'DnCNN':
                out_train = torch.clamp(noisy_image - model(noisy_image), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

        model.eval()  # the end of each epoch
        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
            img_noise_val = img_val + noise
            img_val = Variable(img_val.to(device), volatile=True)
            img_noise_val = Variable(img_noise_val.to(device), volatile=True)
            out_val = model(img_noise_val)
            if opt.model == 'DnCNN':
                out_val = torch.clamp(img_noise_val - model(img_noise_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        # save model
        torch.save(model.state_dict(), os.path.join(opt.out, 'net.pth'))


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
    main()
