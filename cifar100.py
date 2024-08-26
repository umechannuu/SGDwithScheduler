'''Train CIFAR100 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import wandb
import math
from models.wideresnet import WideResNet28_10
from models.resnet import resnet18
from utils import progress_bar
from sgd import SGD





def train(epoch, steps):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    p_norm = 0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        steps += 1
        

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    
        if args.method in ["lr", "linear", "poly"]:
            scheduler.step()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if args.method in ["lr", "cosine", "poly", "linear", "step_lr"]:
            
            if epoch < args.warmup_epochs:
                last_lr = warmup_scheduler.get_last_lr()[0]
            else:
            
                last_lr = scheduler.get_last_lr()[0]
            wandb.log({
                'last_lr': last_lr,
            })
    
    
    p_norm = get_full_grad_list(net,trainset,optimizer)
    
    wandb.log({'fullgrad_norm': p_norm})
    
    if args.method in ["lr", "hybrid", "cosine", "poly", "exp", "linear", "step_lr"]:
        
        if epoch < args.warmup_epochs:
            last_lr = warmup_scheduler.get_last_lr()[0]
        else:
    
            last_lr = scheduler.get_last_lr()[0]
        wandb.log({'lr': last_lr,})
    
        
    

    training_acc = 100.*correct/total
    wandb.log({
        'training_acc': training_acc,
        'training_loss': train_loss / (batch_idx + 1),
    })

    return steps

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

           
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    wandb.log({
        'accuracy': acc
    })

# get fullgradnorm
def get_full_grad_list(net, train_set, optimizer):
    parameters=[p for p in net.parameters()]
    batch_size=128
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    device='cuda:0'
    init=True
    full_grad_list=[]

    for i, (xx,yy) in (enumerate(train_loader)):
        xx = xx.to(device, non_blocking = True)
        yy = yy.to(device, non_blocking = True)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
        loss.backward()
        if init:
            for params in parameters:
                full_grad = torch.zeros_like(params.grad.detach().data)
                full_grad_list.append(full_grad)
            init=False

        for i, params in enumerate(parameters):
            g = params.grad.detach().data
            full_grad_list[i] += (batch_size / len(train_set)) * g
    total_norm = 0.0
    for grad in full_grad_list:
        total_norm += grad.norm().item() ** 2

    return total_norm ** 0.5

def lr_lambda(steps):
    return 1 / math.sqrt(steps+1) 

# warmup by steps
def linear_lr_lambda(steps):
    if steps < warmup_steps:
        return (steps + 1) / warmup_steps
    else:
        return 1 - ((steps - warmup_steps) / (total_steps - warmup_steps))
# warmup by epochs
def warmup_lr_lambda1(epoch):
    if epoch < args.warmup_epochs:
        return (epoch + 1) / args.warmup_epochs
    else:
        return 1
    
        

def norm_work(norm_list, norm):
    norm_list.append(norm)
    average = sum(norm_list) / len(norm_list)
    return average





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=32, type=int, help='training batch size')
    parser.add_argument('--epochs', default=200, type=int, help="the number of epochs")
    parser.add_argument('--decay_epoch', default=40, type=int, help="the number of epochs to decay leraning rate")
    parser.add_argument('--power', default=1.0, type=float, help="polinomial or exponential power")
    parser.add_argument('--warmup_epochs', default=0, type=int, help="the number of epochs for warmup")
    parser.add_argument('--warmup_steps', default=0, type=int, help="the number of steps for a method 'sampling'")
    parser.add_argument('--method', default="batch", type=str, help="constant, lr, poly, cosine, linear")
    parser.add_argument('--model', default="ResNet18", type=str, help="ResNet18, WideResNet")
    parser.add_argument('--steps', default=10000, type=int, help="the number of steps for a method 'sampling'")
    
    args = parser.parse_args()

    

    # wandb setup
    wandb_project_name = "WRITE ME"
    wandb_exp_name = f"CIFAR100,{args.method},b={args.batchsize},lr={args.lr},warmup={args.warmup_epochs},p={args.power}"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,)
    wandb.init(settings=wandb.Settings(start_method='fork'))

    print('==> Preparing data..')
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    device = 'cuda:0'
    if args.model == "ResNet18":
        net = resnet18()
        print("model: ResNet18")
    if args.model == "WideResNet":
        net = WideResNet28_10()
        print("model: WideResNet28_10")
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    
    
    total_steps = (50000 / args.batchsize)*args.epochs
    warmup_steps = args.warmup_steps
    
    print("total_steps: ", total_steps)
    print("warmup_steps: ", warmup_steps)
    
    lr = args.lr
    optimizer = SGD(net.parameters(), lr=lr)

    if args.method == "lr":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.method == "step_lr":   
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.707106781186548)
   
    elif args.method == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200-args.warmup_epochs)
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_lambda1)

    elif args.method == "poly":
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=args.power)
    
    elif args.method == "linear":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_lr_lambda)


    


    
    print(optimizer)
    
    next_batch = args.batchsize
    steps = 0
    for epoch in range(args.epochs):
        steps = train(epoch,steps)
        print("steps: ", steps)
        test(epoch)
        if args.method in ["cosine", "step_lr"]:
            if epoch < args.warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step()

