from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import ResNet50
from utils import Logger
from overfitting_pkg import pgd, adv_loader_data, loader_inference

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=110)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=30)
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--ckpt-name', type=str, default='resnet50')
parser.add_argument('--gpu-id', type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.cuda = not args.no_cuda and torch.cuda.is_available()
sys.stdout = Logger(os.path.join(args.save_dir, args.ckpt_name+'_vanilla.txt'))

torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = datasets.CIFAR10('data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                     ]))
test_dataset = datasets.CIFAR10('data', train=False, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

if args.ckpt_name == 'resnet50':
    model = ResNet50()
if args.cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % args.log_interval == 0:
            print('train epoch {} : [{}/{} ({:.1f}%)]   loss: {:.6f}   acc: {}/{} ({:.2f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                       train_loss / (batch_idx + 1), correct, total, 100. * correct / total))

def test_inference():
    test_loss, psr = loader_inference(model, test_loader)
    print('Standard test acc:{:.2f}%  test loss:{:.6f}'.format(psr, test_loss))

def robust_inference(epoch):
    global best_acc
    adv_samples = []
    target_tensor = []
    model.eval()
    for bi, batch in enumerate(test_loader):
        if bi == 10:
            break
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        crafted, _ = pgd(inputs, targets, model, iters=10)
        adv_samples.append(crafted)
        target_tensor.append(targets)
    adv_samples, targets = torch.cat(adv_samples, 0), torch.cat(target_tensor, 0)
    if args.cuda:
        adv_samples = adv_samples.cpu()
        targets = targets.cpu()
    adv_loader = adv_loader_data(args, adv_samples, targets)
    test_loss, rsr = loader_inference(model, adv_loader)
    print('Robust test acc:{:.2f}%  test loss:{:.6f}'.format(rsr, test_loss))
    state = {'net': model.state_dict(), 'pgd_acc': rsr, 'epoch': epoch}
    torch.save(state, os.path.join('checkpoint', args.ckpt_name + '_vanilla_final.pth'))
    if rsr > best_acc:
        print('Saving current best model..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join('checkpoint', args.ckpt_name+'_vanilla_best.pth'))
        best_acc = rsr


best_acc = 0.0
start_epoch = 1
for epoch in range(start_epoch, args.epochs + 1):
    if epoch in [75, 90, 100]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    test_inference()
    robust_inference(epoch)