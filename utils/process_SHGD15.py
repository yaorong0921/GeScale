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

import os
"""
This is for processing the .csv file into .txt file, which is imported to the dataloader.
"""

with open('labels.csv') as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
with open('category.txt','w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

files_input = ['validation_new.csv','train_new.csv']
files_output = ['val_videofolder_new.txt','train_videofolder_new.txt']
cnt = 0
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    start_idx=[]
    end_idx=[]

    for line in lines:
        line = line.rstrip()
        items = line.split(',')
        folders.append(items[0])
        idx_categories.append(dict_categories[items[1]])
        start_idx.append(items[2])
        end_idx.append(items[3])
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        curStartIDX = int(start_idx[i])
        curEndIDX = int(end_idx[i])

        # counting the number of frames in each video folders
        dir_files_grayscale = os.listdir(os.path.join('./Mydataset/Grayscale', curFolder))
        dir_files_depth = os.listdir(os.path.join('./Mydataset/Depth', curFolder))
        if (len(dir_files_grayscale) == len(dir_files_depth)):
            output.append('%s %d %d %d %d'%(curFolder, (curEndIDX-curStartIDX+1), curIDX, curStartIDX, curEndIDX))
            cnt += 1
        else:
            print(curFolder + 'Wrong!')
            break
        print('%d/%d'%(i, len(folders)))
    with open(filename_output,'w') as f:
        f.write('\n'.join(output))
