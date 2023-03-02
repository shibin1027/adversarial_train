import torch
import torch.nn.functional as F
from torch.autograd import Variable

def inference(model, loader, args, note='None'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(loader.dataset)
    psr = 100. * float(correct) / len(loader.dataset)
    print('<< {} >> Average loss: {:.4f}, Predict Success Rate: {}/{} ({:.2f}%)'.format(
         note, test_loss, correct, len(loader.dataset), psr))
    

def inference_feat(model, loader, args, note='None'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            _, output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(loader.dataset)
    psr = 100. * float(correct) / len(loader.dataset)
    print('<< {} >> Average loss: {:.4f}, Predict Success Rate: {}/{} ({:.2f}%)'.format(
         note, test_loss, correct, len(loader.dataset), psr))
