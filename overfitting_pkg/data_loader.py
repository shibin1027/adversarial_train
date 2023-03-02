from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def clean_loader_cifar(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    data_set = datasets.CIFAR10('../data', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    clean_loader= DataLoader(data_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return clean_loader

def adv_loader_data(args, adv_samples, targets):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    data_set = TensorDataset(adv_samples, targets)
    adv_loader = DataLoader(data_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return  adv_loader











