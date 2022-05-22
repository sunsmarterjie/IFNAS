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
#import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from genotypes import geno_image, geno_image_normal, geno_image_search1, geno_image_search2, geno_image_search3, show_genotype, geno_image_search5, taas_no_sample_l8_keep1, taas_no_sample_l8_keep2, taas_sample_l4, taas_sample_l4_v2
from torch.autograd import Variable
#from model3 import NetworkImageNet as Network
#from model4 import NetworkImageNet as Network
from model5 import NetworkImageNet as Network
#try:
#    from torchprofile import profile_macs
#    print('Installed')
#except:
#    print('Install')
#    strs = 'cd torchprofile $$ python setup.py install'
#    os.system(strs)
#    from torchprofile import profile_macs
import moxing as mox
#try:
#    from thop import profile
#except:
#    os.system('pip install thop')
#    from thop import profile
print('Initial Sleep')
time.sleep(10)
print('Start training')

start = time.time()
# mox.file.copy('s3://bucket-7001/tianyunjie/datasets/imagenet-1000', '/cache/imagenet')
mox.file.copy('s3://bucket-cv-competition-bj4/tianyunjie/datasets/imagenet-1000-tar/imagenet.tar', '/cache/imagenet.tar')
end = time.time()

duration = end - start
print('Coping time: %d s' % duration)
start = time.time()
#cmd = 'cd /cache && tar -xf /cache/imagenet.tar'
cmd = 'cd /cache/ && tar -xvf /cache/imagenet.tar'
os.system(cmd)

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/cache/imagenet', help='cloud train')
# parser.add_argument('--data', type=str, default='/home/ma-user/work/tianyunjie/Imagenet/imagenet-10', help='debug data')
parser.add_argument('--save', type=str, default='/cache/taas_sample_l4_v2_4gpus_normal', help='cloud train')
# parser.add_argument('--save', type=str, default='/cache/cat2sum_bs1024lr05_C56layer14_', help='debug')
##############################################################################################
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='TAAS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler, linear or cosine')
## NVIDIA DALI
parser.add_argument('--dali', action='store_true', default=False, help='if we use NVIDIA DALI')
parser.add_argument('--dali_cpu', action='store_true', default=False, help='if we use NVIDIA DALI CPU')
parser.add_argument('--prefetch', action='store_true', default=False, help='if we use prefetch queue')
# cloud options
parser.add_argument('--data_url',  type=str, default='s3://bucket-cv-competition-bj4/tianyunjie/dataset/imagenet/', help='input_data_url')
parser.add_argument('--train_url', type=str, help='train_dir')
parser.add_argument('--num_gpus', default=0, type=int, help='number of gpus')
parser.add_argument('--tmp_data_dir', type=str, default='/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try_not', help='note for this run')


args, unparsed = parser.parse_known_args()
args.batch_size = args.batch_size * int(torch.cuda.device_count())

if not mox.file.exists(args.save):
    mox.file.make_dirs(args.save)
args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000
# prepare datset
data_dir = os.path.join(args.tmp_data_dir, 'imagenet')
if not mox.file.exists(data_dir):
    start_time = time.time()
    mox.file.copy_parallel(args.data_url, data_dir)
    end_time = time.time()
    duration = end_time - start_time
    print('Processing time: %ds' % duration)

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

    
class PrefetchLoader:

    def __init__(self,
            loader,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
#                next_input = next_input.float().sub_(self.mean).div_(self.std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
    


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
#    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
#    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    #genotype = eval("genotypes.%s" % args.arch)
    show_genotype(taas_sample_l4_v2)
    model = Network(args.init_channels, CLASSES, taas_sample_l4_v2, args.auxiliary)
    model.drop_path_prob = 0.0
#    try:
#        macs = profile_macs(model, torch.randn(1, 3, 224, 224))
#        logging.info("MACS = %fM", macs / 1000000)
#    except:
#        print('MACs profile not installed, pass')
#    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224), ))
#    logging.info("FLOPS = %fM", flops / 1000000)
#    logging.info("PARAMS = %fM", params / 1000000)    
#    if args.parallel:
    model = nn.DataParallel(model)
    model = model.cuda()
#    else:
#        model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
	  
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        #betas=(0.9, 0.999),
        weight_decay=args.weight_decay
        )
    
    '''
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.learning_rate, alpha=0.99, eps=1e-08, 
        weight_decay=args.weight_decay, momentum=0, centered=False)
    '''


    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
    if args.dali:
        dali_whl = os.path.join('/cache', 'nvidia_dali-0.12.0-819496-cp36-cp36m-manylinux1_x86_64.whl')
        if not os.path.exists(dali_whl):
            mox.file.copy('s3://bucket-7001/chenxin/tools/nvidia_dali-0.12.0-819496-cp36-cp36m-manylinux1_x86_64.whl', dali_whl)
            os.system('pip install /cache/nvidia_dali-0.12.0-819496-cp36-cp36m-manylinux1_x86_64.whl')
        from dali_reader import HybridTrainPipe, HybridValPipe
        try:
            from nvidia.dali.plugin.pytorch import DALIClassificationIterator
        except ImportError:
            raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
            
        pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, data_dir=traindir, crop=224, dali_cpu=args.dali_cpu)
        pipe.build()
        train_queue = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
        pipe = HybridValPipe(batch_size=200, num_threads=4, data_dir=validdir, crop=224, size=256)
        pipe.build()
        print(pipe.epoch_size('Reader'))
        valid_queue = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))                                                  
    else:        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
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

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
        if args.prefetch:
            print('Using prefetch loader')
            train_queue = PrefetchLoader(train_loader)
        else:
            train_queue = train_loader

#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=0.0001)
    best_acc_top1 = 0
    best_acc_top5 = 0
    for epoch in range(args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
#        current_lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
            print(optimizer)         
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        if args.dali:
            train_acc, train_obj = train_dali(train_queue, model, criterion_smooth, optimizer)
            train_queue.reset()
        else:
            train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('Train_acc: %f', train_acc)
        
        if args.dali:
            valid_acc_top1, valid_acc_top5, valid_obj = infer_dali(valid_queue, model, criterion)
            valid_queue.reset()
        else:
            valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top5: %f', valid_acc_top5)
        logging.info('Best Valid_acc_top1: %f', best_acc_top1)
        logging.info('Best Valid_acc_top5: %f', best_acc_top5)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)
        
    s3_path = 's3://bucket-cv-competition-bj4/tianyunjie/checkpoints/'
    file_name = args.save.split('/')[-1]
    if not args.note == 'try':
        try:
            mox.file.copy_parallel(args.save, os.path.join(s3_path, file_name)) 
            print('Done..')
        except:
            print('Network is busy. Checkpoint may be lost.')
            mox.file.copy_parallel(args.save, os.path.join(s3_path, file_name))
            print('Done..')
        
def adjust_lr(optimizer, epoch):
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
    #os.system('nvidia-smi')
    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

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

def train_dali(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()
    os.system('nvidia-smi')
    for step, data in enumerate(train_queue):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

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
            logits, _ = model(input)
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
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg

def infer_dali(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, data in enumerate(valid_queue):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        with torch.no_grad():
            logits, _ = model(input)
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
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg



if __name__ == '__main__':
    main() 
