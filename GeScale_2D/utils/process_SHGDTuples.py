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


with open('SHGD13_labels.csv') as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
with open('SHGD13_category.txt','w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i


files_input = 'Tuples.txt'
files_output = 'test_list.txt'
cnt = 0
labels = []
with open(files_input) as f:
    lines = f.readlines()

for line in lines:
    label_one = []
    line = line.rstrip()
    items = line.split(',')
    label_one.append(dict_categories[items[0]])
    label_one.append(dict_categories[items[1]])
    label_one.append(dict_categories[items[2]])
    labels.append(label_one)
output = []
for i in range(len(labels)):

    # counting the number of frames in each video folders
    dir_files_grayscale = os.listdir(os.path.join('../Images/Tuples/set1/Grayscale', str(i)))
    dir_files_depth = os.listdir(os.path.join('../Images/Tuples/set1/Depth', str(i)))
    if (len(dir_files_grayscale) == len(dir_files_depth)):
        output.append('%s %d %d %d %d'%(i, len(dir_files_depth), int(labels[i][0]), int(labels[i][1]), int(labels[i][2])))
    else:
        print(i + 'Wrong!')
        break
    print('%d/%d'%(i, len(labels)))
    with open(files_output,'w') as f:
        f.write('\n'.join(output))
