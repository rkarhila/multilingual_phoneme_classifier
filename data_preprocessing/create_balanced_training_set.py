#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is (was?) in /l/rkarhila/speecon_wsj_phoneme_dnn/data_preprocessing


#
#  1. Load pickled triphone features from disk
#
#  2. Build a balanced set from the data
#
#  3. Calculate normalisation scores and save them to disk
#
#  4. Z-normalise and pickle for immediate use in DNN classification
#

import io
import os
import numpy as np
import subprocess
from subprocess import Popen, PIPE, STDOUT
import re
import math 
import struct
import time
import cPickle
import errno    
import sys

debug=False

corpus = "speecon-adult"


#
#  Feature extraction like this:
#

feature_extraction_script='/l/rkarhila/yeat_another_feature_extractor_for_siak/extract_with_start_end.sh'

#
# A directory that we shouldn't use for anything?
#

audiofilebase = '/l/rkarhila/speecon_wsj_phoneme_dnn/data_preprocessing/triphones/'

#
# Save the final stuff here:
#

statistics_dir = '/l/rkarhila/speecon_wsj_phoneme_dnn/data_preprocessing/statistics/'

pickle_dir='/l/rkarhila/speecon_wsj_phoneme_dnn/data_preprocessing/trip-pickles/'+corpus+'/'



#
# Long vowels and non-vowels, and possible diphtongs:
#

combinations = ['aa','ai','ao','ae',
                'au','ea','ee','ei','eo','eu','ey','eä','ia','ie','ii',
                'io','iu','iy','iä','iö','oa','oe','oi','oo','ou','ua','ue',
                'ui','uo','uu','yi','yy','yä','yö','äe','äi','äy','ää','äö',
                'öi','öy','öä','öö','ng','nn','mm','kk','pp','hh','ll','pp',
                'rr','ss','tt' ]

vowels = [ 'a', 'ä', 'e', 'i', 'o', 'ö', 'u', 'y' ];
nonvow = [ 'd', 'f', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v']


#
# Settings for feature extraction:
#

datatypelength = 2 # 16 bits = 2 bytes, no?

frame_length = 400
frame_step = 128

frame_leftovers = frame_length-frame_step

padding_array = bytearray()

progress_length = 80

max_num_samples=8000 # 0.5 should be enough for any reasonable phoneme, right?

#perc99_length = 11136
#perc99_length_frames = 84

#perc99_length = 8064
#perc99_length_frames = 63

#max_feature_length=perc99_length_frames*feature_dimension

max_num_classes = 10000
feature_dimension=30

max_num_frames=63
max_num_monoclasses = 200


#max_num_samples=100160
assigned_num_samples=100

# tmp directory for feature extraction.
# This should reside in memory (tempfs or whatever it's called, often under /dev/shm/)

tmp_dir="/dev/shm/siak-feat-extract-python-"+str(time.time())
try:
    os.makedirs(tmp_dir)
except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(tmp_dir):
        pass
    else:
        raise   



#
#   Data collection defitinions - train, dev and eval sets:
#



collections = [ {'name' : 'training',
                 'recipe' : '/teamwork/t40511_asr/c/speecon-fi/dataset_division/speecon_train_clean.recipe',
                 'condition' : 'clean',
                 'numlines' : 16254},
                {'name' : 'valid',
                 'recipe' : '/teamwork/t40511_asr/c/speecon-fi/dataset_division/speecon_clean_devel.recipe',
                 'condition' : 'clean',
                 'numlines' : 540},
                {'name' : 'eval',
                 'recipe' : '/teamwork/t40511_asr/c/speecon-fi/dataset_division/speecon_clean_eval.recipe',
                 'condition' : 'clean',
                 'numlines' : 711}
             ]


collections = [ {'name' : 'training-short',
                 'recipe' : '/teamwork/t40511_asr/c/speecon-fi/dataset_division/speecon_train_clean.recipe',
                 'condition' : 'clean',
                 'numlines' : 162},
                {'name' : 'valid-short',
                 'recipe' : '/teamwork/t40511_asr/c/speecon-fi/dataset_division/speecon_clean_devel.recipe',
                 'condition' : 'clean',
                 'numlines' : 54},
                {'name' : 'eval-short',
                 'recipe' : '/teamwork/t40511_asr/c/speecon-fi/dataset_division/speecon_clean_eval.recipe',
                 'condition' : 'clean',
                 'numlines' : 71}
             ]

#collections = [ {'name' : 'python-devel-training',
#                 'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/data_preprocessing/speecon_train_clean.recipe.head10',
#                 'condition' : 'clean',
#                 'numlines' : 16254} ]


featdim1 = -1;
featdim2 = -1;

for collection in [collections[0]]:

    pickle_path=os.path.join(pickle_dir, collection['condition']+"-"+collection['name'])

    classcount=0
    
    samplecounts=np.zeros([1000])

    all_data_dict = {}

    classes = {}

    for picklefile in os.listdir(pickle_path):

        #m = re.search(r'\.([^.]+)\.pkl$', picklefile.encode('utf-8').strip() )
        m = re.search(r'\.([^.]+)\.pkl$', picklefile.strip() )
        mono = m.group(1)

        if mono:
            #print "Loading "+mono
            pickledata = cPickle.load(open(os.path.join(pickle_path,picklefile), 'r'))

            if (featdim1 < 0):
                featdim1 = pickledata['data'].shape[1]
                featdim2 = pickledata['data'].shape[2]
            
            #all_data_dict[mono] = pickledata['data']
            if mono not in classes.keys():
                classes[mono] = classcount

            #print pickledata['data'].shape[0]

            samplecounts[classcount] = pickledata['data'].shape[0]

            #grande_features =  np.zeros([0, max_num_frames, feature_dimension], dtype='float')
            #grande_classes = np.zeros([0, max_num_classes ], dtype='float')


            classcount+=1
    
    samplecounts = samplecounts[0:classcount]

    print "Samplecount %i   Classcount %i   Mean/median samples per class %0.2f / %0.1f    Min samples %i   Max samples %i" % \
        (np.sum(samplecounts), 
         samplecounts.shape[0], 
         np.mean(samplecounts),
         np.median(samplecounts),
         np.min(samplecounts),
         np.max(samplecounts));

    print "capping to min of avg/median"

    clip = math.ceil(max(np.median(samplecounts), np.mean(samplecounts) ))

    samplecounts = np.clip(samplecounts, np.min(samplecounts), clip)
    print "Samplecount %i   Classcount %i   Mean/median samples per class %0.2f / %0.1f    Min samples %i   Max samples %i" % \
        (np.sum(samplecounts), 
         samplecounts.shape[0], 
         np.mean(np.sum(samplecounts)/classcount),
         np.median(samplecounts),
         np.min(samplecounts),
         np.max(samplecounts));


    grande_feature_array = np.zeros([np.sum(samplecounts), featdim1, featdim2])
    grande_class_array = np.zeros([np.sum(samplecounts), max_num_monoclasses])

    sample_counter = 0

    for picklefile in os.listdir(pickle_path):

        m = re.search(r'\.([^.]+)\.pkl$', picklefile.strip() )
        mono = m.group(1)

        if mono:
            #print "Loading "+mono
            pickledata = cPickle.load(open(os.path.join(pickle_path,picklefile), 'r'))

            trclass = classes[mono]

            for i in np.random.permutation(np.arange(0, 
                                                     pickledata['data'].shape[0]))[0:min( clip ,
                                                                                          pickledata['data'].shape[0])]:
                
                grande_feature_array[sample_counter,:,:]=pickledata['data'][i,:,:]
                grande_class_array[sample_counter,trclass] = 1

                sample_counter += 1

    

    new_pickle_dir = os.path.join(pickle_dir, collection['condition']+"_"+str(sample_counter)+"_pickled")

    picklefile = os.path.join(new_pickle_dir,  collection['condition']+"-"+collection['name']+".pkl")

    try:
        os.makedirs(new_pickle_dir)

    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(new_pickle_dir):
            pass
        else:
            raise   

    print "pickling %i items to %s" % ( grande_feature_array.shape[0], picklefile);

    outf = open(picklefile, 'wb')
    
    a = grande_feature_array
    b = grande_class_array

    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]

    np.random.shuffle(c)

    a2 = c[:, :a.size//len(a)].reshape(a.shape)
    b2 = c[:, a.size//len(a):].reshape(b.shape)

    # Pickle the list using the highest protocol available.
    #cPickle.dump({'data': grande_feature_array, 'classes': grande_class_array}, outf, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump({'data': a2, 'classes': b2}, outf, protocol=cPickle.HIGHEST_PROTOCOL)
