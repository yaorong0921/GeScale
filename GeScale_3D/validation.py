'''
/* ===========================================================================
** Copyright (C) 2019 Infineon Technologies AG. All rights reserved.
** ===========================================================================
**
** ===========================================================================
** Infineon Technologies AG (INFINEON) is supplying this file for use
** exclusively with Infineon's sensor products. This file can be freely
** distributed within development tools and software supporting such 
** products.
** 
** THIS SOFTWARE IS PROVIDED "AS IS".  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
** OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
** MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
** INFINEON SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR DIRECT, INDIRECT, 
** INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, FOR ANY REASON 
** WHATSOEVER.
** ===========================================================================
*/
'''

import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy

def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    #conf_mat = torch.zeros(opt.n_finetune_classes, opt.n_finetune_classes)
    #output_file = []
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        ### print out the confusion matrix
        #_,pred = torch.max(outputs,1)
        #for t,p in zip(targets.view(-1), pred.view(-1)):
        #    conf_mat[t,p] += 1

        ### output the file with all the probability
        #out_softmax = torch.nn.functional.softmax(outputs,dim=1,dtype=torch.float16)
        #output_file.extend(out_softmax.data.cpu().tolist())
        
        losses.update(loss.data, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
              epoch,
              i + 1,
              len(data_loader),
              batch_time=batch_time,
              data_time=data_time,
              loss=losses,
              top1=top1,
              top5=top5))
    #print(conf_mat)

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})
### output the file with all the probability
    #with open('output.txt','w') as f:
    #    f.write(str(output_file))
    return losses.avg.item(), top1.avg.item()
