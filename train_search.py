import os
import sys
import numpy as np
import time
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search import Network
from loss import discretization_2operator
from genotypes import show_genotype, supernet_genotype, filter_genotype, sample_geno, TAAS_sample
import moxing as mox

mox.file.copy_parallel('s3://bucket-7001/tianyunjie/datasets/imagenet-100/', '/cache/imagenet')

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/cache/imagenet', help='cloud train')
# parser.add_argument('--data', type=str, default='/home/ma-user/work/tianyunjie/Imagenet/imagenet-10', help='debug data')
parser.add_argument('--save', type=str, default='/cache/original_pcdarts_image1000_bs1024lr05_Cosine',
                    help='cloud train')
# parser.add_argument('--save', type=str, default='/cache/cat2sum_bs1024lr05_C56layer14_', help='debug')
##############################################################################################
parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--arch_learning_rate', type=float, default=.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--arch_weight_decay', type=float, default=3e-3, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--pretrain', type=int, default=50, help='total number of layers')
parser.add_argument('--D_pretrain', type=int, default=65, help='total number of layers')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--threshold', type=float, default=0.1, help='threshold to crop connection when shrink supernet')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
## NVIDIA DALI
parser.add_argument('--dali', action='store_true', default=False, help='if we use NVIDIA DALI')
parser.add_argument('--dali_cpu', action='store_true', default=False, help='if we use NVIDIA DALI CPU')
parser.add_argument('--prefetch', action='store_true', default=False, help='if we use prefetch queue')
# cloud options
parser.add_argument('--data_url', type=str, default='s3://bucket-2707/chenxin/dataset/imagenet/', help='input_data_url')
parser.add_argument('--train_url', type=str, help='train_dir')
parser.add_argument('--num_gpus', default=0, type=int, help='number of gpus')
parser.add_argument('--tmp_data_dir', type=str, default='/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args, unparsed = parser.parse_known_args()

args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 100

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


length = 8

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
# criterion_D = discretization_criterion()
# criterion_D = criterion_D.cuda()
# criterion_D = discretization_goldnas()
# criterion_D = criterion_D.cuda()
criterion_D = discretization_2operator(length=length)
criterion_D = criterion_D.cuda()


def main():
    supernet = supernet_genotype(18, 20, 18, length=length)
    # supernet = geno
    show_genotype_logging(supernet)
    model = Network(args.init_channels, CLASSES, supernet, func='sigmoid', length=length)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model = search(model, criterion)

    supernet = update_supernet(model, supernet, supernet, args.threshold)
    supernet = filter_genotype(supernet)
    logging.info('supernet now:')
    show_genotype_logging(supernet)


def search(model, criterion):
    optimizer_alpha = torch.optim.SGD(
        model.parameter_alpha(0),
        lr=args.arch_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer_omega = torch.optim.SGD(
        model.parameter_omega(args.learning_rate),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    data_dir = os.path.join(args.data)
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
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

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_omega, float(args.epochs))

    for epoch in range(args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer_omega, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if num_gpus > 1:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion, optimizer_omega, optimizer_alpha, epoch)
        logging.info('Train_acc: %f', train_acc)

        # valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        # logging.info('Valid_acc_top1: %f', valid_acc_top1)
        # logging.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        show_alphas(model)
        logging.info('Epoch time: %ds.', epoch_duration)

    return model


def train(train_queue, model, criterion, optimizer_omega, optimizer_alpha, epoch):
    objs = utils.AvgrageMeter()
    objs_D1 = utils.AvgrageMeter()
    objs_D2 = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        if epoch == args.pretrain and step == 0:
            # logging.info('set alphas LR now ... ')
            set_alphas_lr(optimizer_alpha)

        optimizer_alpha.zero_grad()
        optimizer_omega.zero_grad()
        logits = model(input, turns=epoch % (length + 1))
        loss = criterion(logits, target)
        loss_D1, loss_D2 = criterion_D(model, epoch=epoch, pretrain=args.D_pretrain, turns=epoch % (length + 1))
        # loss_D = 0
        loss_ = loss + loss_D1 + loss_D2
        loss_.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer_omega.step()
        optimizer_alpha.step()
        optimizer_alpha.zero_grad()
        optimizer_omega.zero_grad()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        objs_D1.update(loss_D1.item(), n)
        objs_D2.update(loss_D2.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        if step % args.report_freq == 0:
            logging.info('train %03d loss:%e loss_D1:%e loss_D2:%e %f %f', step, objs.avg, objs_D1.avg, objs_D2.avg,
                         top1.avg, top5.avg)

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
            logits, _ = model(input), None
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


def update_supernet(model, geno, supernet, threshold):
    '''
    Inputs:
        alphas have been updated, which decided which connections in supernet to be cropped.
        geno are childnet have been sampled, which corresponds to alphas
        supernet are genotype saved the existing connections;
        threshold decided which connections to be cropped.
    '''
    alphas = [w for w in model.weight_edge]
    for k, stage in enumerate(geno):
        alpha_stage = alphas[k]
        for l, n in enumerate(stage.items()):
            alpha_node = alpha_stage[int(n[0][4:]) - 1]
            try:
                mask = alpha_node > threshold
            except TypeError:
                mask = [alpha_node[0] > threshold]
            delete = []
            delete_flag = 0
            if sum(mask) == 0 and len(mask) > 1:
                max_index = list(alpha_node).index(max(list(alpha_node)))
                mask[max_index] = True
            for i in range(len(n[1])):
                if not mask[i] and len(supernet[k][n[0]]) > 1 and len(geno[k][n[0]]) > 1:
                    delete.append([n, i - delete_flag])
                    delete_flag += 1
            for delete_node in delete:
                supernet[k][delete_node[0][0]].remove(supernet[k][delete_node[0][0]][delete_node[1]])
    return supernet


def show_genotype_logging(geno):
    for i in range(3):
        logging.info('stage%d:' % (i + 1))
        for j, n in enumerate(geno[i].items()):
            logging.info("     %s" % str(n))


def show_alphas(model):
    model.generate_weight()
    logging.info('alphas & betas:')
    for k, w in enumerate(model.weight1_edge):
        logging.info('   A1: stage1\n %d    %s' % (k + 1, str(w.detach())))
        logging.info('   B1: stage1\n %d    %s\n' % (k + 1, str(model.weight1_op[k].detach())))
    logging.info("******************************************************")
    for k, w in enumerate(model.weight2_edge):
        logging.info('   A1: stage2\n %d    %s' % (k + 1, str(w.detach())))
        logging.info('   B1: stage2\n %d    %s\n' % (k + 1, str(model.weight2_op[k].detach())))
    logging.info("******************************************************")
    for k, w in enumerate(model.weight3_edge):
        logging.info('   A1: stage3\n %d    %s' % (k + 1, str(w.detach())))
        logging.info('   B1: stage3\n %d    %s\n' % (k + 1, str(model.weight3_op[k].detach())))
    logging.info("******************************************************")


def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    lr = args.learning_rate * (args.epochs - epoch) / args.epochs
    for k, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr


def set_alphas_lr(optimizer):
    for k, param in enumerate(optimizer.param_groups):
            param['lr'] = args.arch_learning_rate
#     assert False


if __name__ == '__main__':
    main()
