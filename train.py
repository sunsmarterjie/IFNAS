import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from genotypes import geno_image, show_genotype, geno_image_short, geno_image_short2, geno_image_long, geno_image_short4, geno_cosine_similarity, geno_cosine_similarity2
from torch.autograd import Variable
#from model import NetworkImageNet as Network
from model_cosine_similarity import NetworkImageNet as Network
from count_flops import get_model_complexity_info
import moxing as mox

mox.file.copy_parallel('s3://bucket-7001/tianyunjie/datasets/imagenet-100/', '/cache/data/')

'''
start = time.time()
mox.file.copy('s3://bucket-7001/chenxin/dataset/imagenet_tar/imagenet.tar', '/cache/data/imagenet.tar')
end = time.time()
duration = end - start
print('Coping time: %d s' % duration)
start = time.time()
cmd = 'cd /cache/data && tar -xvf /cache/data/imagenet.tar'
os.system(cmd)
cmd = 'cd /cache && ls'
os.system(cmd)
cmd = 'cd /cache/data && ls'
os.system(cmd)
end = time.time()
duration = end - start
print('X time: %d s' % duration)
'''

############################################################################################################################
parser = argparse.ArgumentParser("training imagenet")
#parser.add_argument('--data', type=str, default='/cache/data/imagenet', help='cloud train')
parser.add_argument('--data', type=str, default='/cache/data', help='cloud train')
# parser.add_argument('--data', type=str, default='/home/ma-user/work/tianyunjie/Imagenet/imagenet-10', help='debug data')
parser.add_argument('--save', type=str, default='/cache/darts_original_image100_bs512lr03_', help='cloud train')
# parser.add_argument('--save', type=str, default='./darts_original_image100', help='debug')\
############################################################################################################################
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler, linear or cosine')
parser.add_argument('--note', type=str, default='try', help='note for this run')
#############################################################################################################################
parser.add_argument('--num_gpus', help='comma separated list of GPU(s) to use.', type=int, default=2)
parser.add_argument('--data_url', help='dataset url.', type=str, default='s3://bucket-8280/data/imagenet/')
parser.add_argument('--train_url', help='trainlog url', type=str, default='./')
parser.add_argument('--init_method', help='init')
args, unparsed = parser.parse_known_args()
#############################################################################################################################
args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 100


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    print('---------Genotype---------')
    show_genotype(geno_cosine_similarity2)
    print('--------------------------')
    #model = Network(args.init_channels, CLASSES, auxiliary=args.auxiliary, genotype=geno_image_short4)
    model = Network(geno_cosine_similarity2, args.init_channels, 100)
    flops, params = get_model_complexity_info(model, (224, 224), as_strings=True, print_per_layer_stat=False)
    logging.info('>>>>>:', flops, params)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    #     data_dir = os.path.join(args.data, 'imagenet')
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc_top1 = 0
    best_acc_top5 = 0
    lr = args.learning_rate
    for epoch in range(args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('Train_acc: %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        # utils.save_checkpoint({
        #    'epoch': epoch + 1,
        #    'state_dict': model.state_dict(),
        #    'best_acc_top1': best_acc_top1,
        #    'optimizer' : optimizer.state_dict(),
        #    }, is_best, args.save)


def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
      
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        
        if step % 400 == 0:
            distance = model.cosine_similarity(input)
            for d in distance:
                print(d)
            print()
        
        b_start = time.time()
        optimizer.zero_grad()
        #logits, logits_aux = model(input)
        logits = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            #print(logits_aux.shape)
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                         step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            #logits, _ = model(input)
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg,
                         duration)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
    mox.file.copy_parallel(args.save, 's3://bucket-7001/tianyunjie/logs/')