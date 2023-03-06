from __future__ import print_function
import argparse
import os
import numpy as np
import torch
from models import Feat1_ResNet50
from robust_pkg.adv_attack import fgsm_feat, pgd_feat, cw_feat
from robust_pkg.data_loader import clean_loader_cifar, adv_loader_data
from robust_pkg.inference_tools import inference_feat

parser = argparse.ArgumentParser()
parser.add_argument('--test-batch-size', type=int, default=128)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--eps', type=float, default=0.03)
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--num-cls', type=int, default=10)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

args.model_checkpoint = './checkpoint/resnet50f1_sp30_final.pth'
print("Evaluating...:", args.model_checkpoint)
print('current epsilon:', args.eps)

def craft_adv_samples(data_loader, model, args, attack_method):
    adv_samples = []
    target_tensor = []
    l2_list = []
    model.eval()
    for bi, (inputs, targets) in enumerate(data_loader):
        if bi==20:
            break
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        if attack_method == 'fgsm':
            crafted, l2 = fgsm_feat(inputs, targets, model, num_cls=args.num_cls, eps=args.eps)
        elif attack_method == 'pgd10':
            crafted, l2 = pgd_feat(inputs, targets, model, num_cls=args.num_cls, eps=args.eps, iters=10)
        elif attack_method == 'pgd20':
            crafted, l2 = pgd_feat(inputs, targets, model, num_cls=args.num_cls, eps=args.eps, iters=20)
        elif attack_method == 'pgd50':
            crafted, l2 = pgd_feat(inputs, targets, model, num_cls=args.num_cls, eps=args.eps, iters=50)
        elif attack_method == 'cw10':
            crafted, l2 = cw_feat(inputs, targets, model, num_cls=args.num_cls, eps=args.eps, iters=10)
        elif attack_method == 'cw20':
            crafted, l2 = cw_feat(inputs, targets, model, num_cls=args.num_cls, eps=args.eps, iters=20)
        else:
            raise NotImplementedError
        adv_samples.append(crafted)
        target_tensor.append(targets)
        l2_list.append(l2)

    return torch.cat(adv_samples, 0), torch.cat(target_tensor, 0), sum(l2_list)/len(l2_list)


def main():
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    clean_loader = clean_loader_cifar(args)
    model = Feat1_ResNet50()
    if args.cuda:
        model.cuda()
    checkpoint = torch.load(args.model_checkpoint)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint)

    inference_feat(model, clean_loader, args, note='natural')

    for attack_method in ['fgsm', 'pgd10']:
        adv_samples, targets, l2_mean = craft_adv_samples(clean_loader, model, args, attack_method)
        if args.cuda:
            adv_samples = adv_samples.cpu()
            targets = targets.cpu()
        adv_loader = adv_loader_data(args, adv_samples, targets)

        inference_feat(model, adv_loader, args, note=attack_method)

if __name__ == '__main__':
    main()
