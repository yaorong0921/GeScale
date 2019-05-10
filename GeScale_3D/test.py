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
import time
import math
import numpy as np
import copy

from utils import AverageMeter
from collections import deque


def viterbi_search(inputs,seq):
    """
    Args:
        inputs: list of the probs of every t.
        seq: the number of label changing in one sequence.
    Attributes:
        The most possible path
    """
    L = len(inputs)   ## the number of t
    N = len(inputs[0]) ## the number of classes
    # initialize with the first t
    path = []        # this is the path list
    for m in [0,1,2,6,7,8,9,10,11,12]:
        path.append([m])
    p = inputs[0]                   # this is the probability list
    s = np.zeros(10,dtype=np.uint8) # this is the transition times list
    limit = 300

    for t in range(1, L):
        path_tmp = []  # tmp marks the list during each t for comparison
        p_tmp = []
        s_tmp = []
        for l in range(len(path)):
            if s[l] < seq: # now can change the state and has N choices
                for n in [0,1,2,6,7,8,9,10,11,12]:
                    path_current = copy.deepcopy(path[l])  # the last state of this path
                    state_last = path_current[-1]
                    if state_last != n:
                        path_current.append(n)
                        p_tmp.append(p[l]+inputs[t][n]-0.2)
                    else:
                        p_tmp.append(p[l]+inputs[t][n])
                    path_tmp.append(path_current)
                    s_tmp.append(s[l]+(state_last!=n))
            else: # now can only hold onto the state as last state
                path_current = copy.deepcopy(path[l])  # the last state of this path
                state_last = path_current[-1]
                path_tmp.append(path_current)
                p_tmp.append(p[l]+inputs[t][state_last])
                s_tmp.append(s[l])


    # to get the top 'limit' path
        idxs = sorted(range(len(p_tmp)), key=lambda k: p_tmp[k],reverse=True)
        path = []
        p = []
        s =[]
        for i in range(min(len(p_tmp),limit)):
            index = idxs[i]
            path.append(path_tmp[index])
            p.append(p_tmp[index])
            s.append(s_tmp[index])
    # return the maximal one path with seq times changes
    idx = s.index(seq)
    return path[idx]


def Detector(prob_q,prob_th,start):
    probs = 0

    for d in range(len(prob_q)):
        if start:
            probs += prob_q[d][4]
        else:
            probs += prob_q[d][3]
    if probs > prob_th:
        return True
    else:
        return False

def test(data_loader, model, opt):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()

##### some config that can be changed
    window_len = 5
    window_stride = 5
    n_stride = 1
    detector_len = 8
    sog_th = 5
    eog_th = 5

##### initial all the errors
    SoG_error = 0
    EoG_error = 0
    detector_error = 0
    order_error = 0
    classifier_error = 0

    for i, (inputs, targets, input_length) in enumerate(data_loader):

        step = math.ceil((input_length - opt.sample_duration) / n_stride)
        logits = deque(maxlen=window_len)
        SoG = False
        EoG = False
        detector = deque(maxlen=detector_len)
        input_viterbi = []
        counter = 0

        for j in range(0, step+1):
            M = list(range(j * n_stride, min(j*n_stride+opt.sample_duration, input_length)))
            if len(M) < 8:
                break
            else:
                input_single = inputs[:,:,M[0]:(M[0]+opt.sample_duration),:,:]
                output_single = torch.squeeze(model(input_single))

                detector.append(output_single.softmax(0).data)
                if not SoG:
                    SoG = Detector(detector,sog_th,True) ## continue detecting Start-of-Gesture

                else: ## enter Classifier Queue mode
                    EoG = Detector(detector,eog_th,False) # detecting End-of-Gesture
                    if EoG:  ## End-of-Gesture detected
                        break

                    else:
                        logits.append(output_single.data)
                        counter += 1
                        if (len(logits) == window_len) and (counter%window_stride == 0):
                        ######   activate the window to calculate the new probs

                            weighted_logits = sum(logits)
                            window_output = weighted_logits.softmax(0)
                            _,pred = torch.max(window_output, 0)
                            if not (pred == 3 or pred == 4 or pred == 5):
                            #### signal gestures are ignored
                                input_viterbi.append(window_output)

    ## Deactivate Classifier Queue and start Viterbi decoder
        if len(input_viterbi) > 3:
            path_m = viterbi_search(input_viterbi, 2)
            if not path_m == targets and EoG:
                order_error += 1
                path_m_arr = np.asarray(path_m)
                targets_arr = np.asarray(targets)
                classifier_error += (3-(path_m_arr == targets_arr).sum())
            ###### print the number of tuple, ground truth and the wrong answer
                print(i,targets, path_m)

        else:
            detector_error += 1
        batch_time.update(time.time() - end_time)
        end_time = time.time()

######  monitor the error changes for every tuple
        print('tuple: {0}, processing time: {batch_time.avg:.5f}, detecotr_error: {1}, order_error: {2}'.format(i, detector_error, order_error, batch_time=batch_time))

###### print the summery of one set (810 gesture tuples) in the end
    print(EoG_error, detector_error, order_error,classifier_error)
