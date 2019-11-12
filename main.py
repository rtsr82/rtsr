from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net, SRResNet, VDSR, SRCNN, EDSR
from data import get_training_set, get_test_set

# Training settings
#default Setting for Train
# python main.py --upscale_factor 2 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001 --cuda --sr_run 1
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor", default=2)
parser.add_argument('--upscale_factor', type=int, help="super resolution upscale factor", default=2)
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?', default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--sr_run', type=int, default=1, help='sr repeated No.')
parser.add_argument('--lr_dec', type=float, default=0.9, help='Learning Rate Decrease Ratio per a Epoch')
parser.add_argument('--network', type=str, default='rtsr', help='Network Setlection')

opt = parser.parse_args()


print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor, opt.sr_run)
test_set = get_test_set(opt.upscale_factor, opt.sr_run)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if (opt.network == 'rtsr') :
    print("network : RTSR")
    model = Net(upscale_factor=opt.upscale_factor).to(device)
elif (opt.network == 'srresnet') :
    print("network : SRResNet")
    model = SRResNet().to(device)
elif (opt.network == 'srcnn') :
    print("network : SRCNN")
    model = SRCNN(upscale_factor=opt.upscale_factor).to(device)
elif (opt.network == 'edsr') :
    print("network : EDSR")
    model = EDSR().to(device)

criterion = nn.MSELoss()

model = nn.DataParallel(model)


def train(epoch):
    epoch_loss = 0
    adaptive_lr = adjust_learning_rate(opt, epoch)
    print("Epoch[{}] : Set Learning Rate as {:.5f}".format(epoch, adaptive_lr))
    optimizer = optim.Adam(model.parameters(), adaptive_lr)
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if(iteration%100 == 0) :
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))


    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def adjust_learning_rate(opt, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.lr_dec ** (epoch-1)) # // 2))
    return lr

def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)

