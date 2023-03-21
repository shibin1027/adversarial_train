from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def clean_loader_cifar(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    dataset = datasets.CIFAR10('/media/hdd/msb3/VMF/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    clean_loader= DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return clean_loader

def clean_loader_cifar100(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    dataset = datasets.CIFAR100('/media/hdd/msb3/VMF/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    clean_loader= DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return clean_loader

def clean_loader_svhn(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    dataset = datasets.SVHN('/home/lorne/shibin/adversarial_train/data', split='test', transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    clean_loader= DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return clean_loader

def adv_loader_data(args, adv_samples, targets):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    adv_loader = DataLoader(TensorDataset(adv_samples, targets), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return  adv_loader











