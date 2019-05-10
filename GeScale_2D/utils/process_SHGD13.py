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
The three signal gestures are added. Details are below.
"""

with open('labels_for_tuples.csv') as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
with open('category_for_tuples.txt','w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i


files_input = ['validation_new.csv','train_new.csv']
files_output = ['vallist_for_tuples.txt','trainlist_for_tuples.txt']
cnt = 0
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    start_idx=[]
    end_idx=[]

    for i,line in enumerate(lines):
        line = line.rstrip()
        items = line.split(',')
        label = items[1]
        begin = int(items[2])
        end = int(items[3])
        for j in categories:
            if label == j:
                folders.append(items[0])
                idx_categories.append(dict_categories[label])
                start_idx.append(begin)
                end_idx.append(end)

                # put 16-6 frames before as "Hand Up"
                folders.append(items[0])
                idx_categories.append(dict_categories['Hand Up'])
                start_idx.append((begin-16))
                end_idx.append((begin-6))
                # put 16-6 frames after as "Hand down", and 32 to 17 frames after as "No Gesture"
                if (end+16 <= 1499):
                    folders.append(items[0])
                    idx_categories.append(dict_categories['Hand Down'])
                    start_idx.append(end+6)
                    end_idx.append((end+16))
                    if (end+ 32) <= 1499:
                        folders.append(items[0])
                        idx_categories.append(dict_categories['No Gesture'])
                        start_idx.append(end+17)
                        end_idx.append(end+32)
                # put 32 to 17 frames before as "No Gesture"
                if (begin-32) >= 0:
                    folders.append(items[0])
                    idx_categories.append(dict_categories['No Gesture'])
                    start_idx.append(begin-32)
                    end_idx.append(begin-17)
                else:
                    folders.append(items[0])
                    idx_categories.append(dict_categories['No Gesture'])
                    start_idx.append(0)
                    end_idx.append((begin-17))

                break

    output = []

    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        curStartIDX = int(start_idx[i])
        curEndIDX = int(end_idx[i])

        output.append('%s %d %d %d %d'%( curFolder, (curEndIDX-curStartIDX+1), curIDX, curStartIDX, curEndIDX))
        cnt += 1

        print('%d/%d'%(i, len(folders)))
    with open(filename_output,'w') as f:
        f.write('\n'.join(output))
