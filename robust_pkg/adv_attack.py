import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def fgsm(img, label, model, num_cls=10, eps=0.007):
    criterion = F.cross_entropy
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    out = model(adv)
    loss = criterion(out, label)
    loss.backward()
    noise = adv.grad
    adv.data = adv.data + eps * noise.sign()
    adv.data.clamp_(0.0, 1.0)

    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()
    return adv.detach(), l2


def pgd(img, label, model, num_cls=10, eps=0.03, iters=10,step=0.007):
    criterion = F.cross_entropy
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        out_adv = model(adv)
        loss = criterion(out_adv, label)
        loss.backward()
        noise = adv.grad
        adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2


def one_hot_tensor(y_batch_tensor, num_classes):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0), num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce

    def forward(self, logits, targets):
        onehot_targets = one_hot_tensor(targets, self.num_classes)
        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]
        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))
        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num
        return loss

def cw(img, label, model, num_cls=10, eps=0.03, iters=20, step=0.007):
    criterion = CWLoss(num_classes=num_cls)
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        out_adv = model(adv)
        loss = criterion(out_adv, label)
        loss.backward()
        noise = adv.grad
        adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2



def fgsm_feat(img, label, model, num_cls=10, eps=0.007):
    criterion = F.cross_entropy
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    _, out = model(adv)
    loss = criterion(out, label)
    loss.backward()
    noise = adv.grad
    adv.data = adv.data + eps * noise.sign()
    adv.data.clamp_(0.0, 1.0)

    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()
    return adv.detach(), l2


def pgd_feat(img, label, model, num_cls=10, eps=0.03, iters=10,step=0.007):
    criterion = F.cross_entropy
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        _, out_adv = model(adv)
        loss = criterion(out_adv, label)
        loss.backward()
        noise = adv.grad
        adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2


def cw_feat(img, label, model, num_cls=10, eps=0.03, iters=20, step=0.007):
    criterion = CWLoss(num_classes=num_cls)
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        _, out_adv = model(adv)
        loss = criterion(out_adv, label)
        loss.backward()
        noise = adv.grad
        adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2