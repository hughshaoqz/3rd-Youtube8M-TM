"""
Ensemble submission files.
"""

import os
import csv
import pandas as pd # not key to functionality of kernel
import glob

import sys
import csv
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


SUBDIR_SINGLE_MODEL = '/home/hugh/Projects/competition/kaggle-youtube/submissions/single_ensemble_model/'
SUBDIR_ENSEMBLE_MODEL = '/home/hugh/Projects/competition/kaggle-youtube/submissions/ensemble_model/'

sub_files = glob.glob(SUBDIR_SINGLE_MODEL+'*')

# Weights of the individual subs
sub_weight = [(int(sub_file[-7:-4])/1000)**2 for sub_file in sub_files]


Hlabel = 'Class' 
Htarget = 'Segments'
npt = 100000 # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = ( 1 / (i + 1) )
    
print(place_weights)

lg = len(sub_files)
sub = [None]*lg
for i, file in enumerate( sub_files ):
    ## input files ##
    print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file,"r"))
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

## output file ##
out = open(os.path.join(SUBDIR_ENSEMBLE_MODEL, 'ensemble_v3.csv'), "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel,Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()