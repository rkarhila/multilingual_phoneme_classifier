#!/usr/bin/env python
# -*- coding: utf-8 -*-


# This file is (was?) in /l/rkarhila/speecon_wsj_phoneme_dnn/data_preprocessing

#
#  1. Fetch lists of all speecon data (of a given condition, for example: "clean")
#
#  2. Preprocess the alignment files, joining probable diphtongs
#
#  3. Divide each speecon data files into single phoneme chunks
#
#  4. Run the chunks through feature extraction shell script
#
#  5. Store the features and their associated phoneme information in arrays
#
#  6. Pickle for future normalisation (with other corpora) 
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

max_num_frames=40 #63
max_num_monoclasses = 100


#max_num_samples=100160
assigned_num_samples=100
file_batch_size=2000

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
                  'numlines' : 1625},
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


# In the recipe files, look for audio like:
#
# audio=/work/asr/c/speecon-fi/adult/ADULT1FI/BLOCK00/SES002/SA002CC1.FI0
#
# and transcriptions like: 
#
# alignment=/work/asr/c/speecon-fi/adult/alignment/triphone/SES002/SA002CC1.phn
#
# and change them to
#
# /teamwork/t40511_asr/c/speecon-fi/adult/ADULT1FI/BLOCK00/SES002/SA002CC1.FI0
# and
# /teamwork/t40511_asr/c/speecon-fi/adult/alignment/triphone-state/BLOCK00/SES002/SA002CC1.phn
#
# and process:




print "start!"

triphoneclasses = {}
classcounter=0

for collection in collections:


    triphonedata = {}
    #triphonecounter = {}

    #features=np.zeros([assigned_num_samples, max_num_frames, feature_dimension], dtype='float32')
    #classes = np.zeros([assigned_num_samples, max_num_classes], dtype='float')

    recipefile = open( collection['recipe'] , 'r')
    recipefilecounter = 0    
    too_long_counter = 0
    all_trips_counter = 0

    tmpfilecounter = 0

    progress_interval = math.ceil(collection['numlines']/1000.0)

    statistics_file=statistics_dir+"/"+corpus+"-"+collection['condition']+"-"+collection['name']+".triphone-frame-counts"
    statistics_handle = open(statistics_file, 'w')

    class_file=statistics_dir+"/"+corpus+"-"+collection['condition']+"-"+collection['name']+".triphone-classes"
    class_handle= open(class_file, 'w')

    phone_merge_file=statistics_dir+"/"+corpus+"-"+collection['condition']+"-"+collection['name']+".phone-merge"
    phone_merge_handle = open(phone_merge_file, 'w')
    for r in recipefile.readlines():
        
        audiofile = re.sub('audio=/work/asr', '/teamwork/t40511_asr',  re.findall('audio=/work/asr/c/speecon-fi/adult/ADULT1FI[^ ]+', r)[0])
        if (collection['name'][0:8] == 'training') or (collection['name'] == 'python-devel-training'):
            labelfile = re.sub(r'alignment=/work/asr/c/speecon-fi/adult/alignment/triphone/SES(..)', r'/teamwork/t40511_asr/c/speecon-fi/adult/alignment/triphone-state/BLOCK\1/SES\1', re.findall('alignment=/work/asr/c/speecon-fi/adult/alignment[^ ]+', r)[0])
        else:
            labelfile = re.sub(r'alignment=/work/asr/c/speecon-fi/adult/alignment/triphone/', r'/teamwork/t40511_asr/c/speecon-fi/adult/alignment/triphone-state/', re.findall('alignment=/work/asr/c/speecon-fi/adult/alignment[^ ]+', r)[0])

        if debug:
            print "Labelfile %i/%i: %s" % (recipefilecounter, collection['numlines'], labelfile)
            print "Audiofile %i/%i: %s" % (recipefilecounter, collection['numlines'], audiofile)

        with io.open(labelfile ,'r',encoding='iso-8859-15') as f:
            
            new_align = []

            current_start = 0
            current_end = 0
            current_model = False
            current_premodel = False
            current_postmodel = False

            skip = False

            phonect = 0
            statect = 0
            
            lcounter = 0

            # For printing the phoneme sequences into a log:
            labelstring=''
            skipmark=False

            startmark=-1
            endmark = -1

            phone={}
            olderphone= {'premodel': '__', 'model': '__', 'postmodel':'__'}

            for l in  f.readlines():

                # If we have a short pause model:
                #if '+' not in l:
                #    no_skipping = True
                #    skipmark = True
                
                # We'll process the label line by line with a two-phone delay:

                if '+' in l:
                    #print "Looking at %s"%(l)
                    [start, end, premodel, model, postmodel, state] = re.split(r'[ .+-]', l.encode('utf-8').strip() )
                    
                else:
                    [start, end, model, state] = re.split(r'[ .+-]', l.encode('utf-8').strip() )
                    premodel = '_'
                    postmodel = '_'
                    if debug:
                        if model == '_':
                            print "here is a pause"
                        elif model == '__':
                            print "here is a begin/end silence"
                                                        

                    skipmark = True

                if 1==1:

                    #print [start, end, premodel, model, postmodel, state]
                    if state=='0':
                        if debug:
                            print "here is %s-%s+%s.%s ; current olderphone is %s-%s+%s" % (premodel, model, postmodel, state,
                                                                                            olderphone['premodel'], 
                                                                                            olderphone['model'],
                                                                                            olderphone['postmodel'])
                            
                        phone = {'start':start, 
                                 'premodel':premodel, 
                                 'model': model,
                                 'postmodel':postmodel,
                                 'state':state }

                    if state=='2':
                        phone['end'] = end
                        
                        if (not skipmark) and ((olderphone['model']+olderphone['postmodel']) in combinations):
                            if debug:
                                print "Merging two models: %s-%s+%s and %s-%s+%s => %s-%s+%s " %  \
                                (olderphone['premodel'], olderphone['model'],olderphone['postmodel'], 
                                 phone['premodel'], phone['model'],phone['postmodel'],
                                 olderphone['premodel'], (olderphone['model']+olderphone['postmodel']), phone['postmodel'])

                            olderphone['postmodel'] = phone['postmodel']
                            olderphone['model'] = (olderphone['model']+phone['model'])
                            olderphone['end'] = phone['end']

                            if debug:
                                print " --- current olderphone is %s-%s+%s" % (olderphone['premodel'], 
                                                                               olderphone['model'],
                                                                               olderphone['postmodel'])
                        else:     
                            if (olderphone['model'] != '__'):
                                if debug:
                                    print "saving %s-%s+%s " %  (olderphone['premodel'], olderphone['model'],olderphone['postmodel'])
                                new_align.append({'pre' : olderphone['premodel'],
                                                  'model' : olderphone['model'],
                                                  'post' : olderphone['postmodel'],
                                                  'start' : olderphone['start'],
                                                  'end' : olderphone['end'],
                                                  'triphone': "%s-%s+%s" % (olderphone['premodel'] , olderphone['model'], olderphone['postmodel']),
                                                  'sortable': "%s--%s++%s" % (olderphone['model'] , olderphone['premodel'], olderphone['postmodel'])
                                              })
                                labelstring += '.'+olderphone['model']
                                if skipmark:
                                    labelstring += '. '

                            olderphone = phone

                            
                            
                        skipmark=False

                
            

                # if lcounter > 1:                    
                    
                #     if '+' not in olderl:
                #         [start, end, model, state] = re.split(r'[ .+-]', olderl.encode('utf-8').strip() )
                #         premodel = '_'
                #         postmodel = '_'
                #     else:
                #         [start, end, premodel, model, postmodel, state] = re.split(r'[ .+-]', olderl.encode('utf-8').strip() )

                #     #print "state: "+state+", skip: "+str(skip)
                #     if (state == '0') and not skip:
                #         #print "start "+str(start) + " prev model >"+ str(premodel) + "< this model >" + str(model) + "< next model >"+ str(postmodel)+"<"
                #         current_start = start
                #         current_postmodel = postmodel
                #         current_premodel = premodel

                #         if (model+postmodel) in combinations:
                #             if not no_skipping:
                #                 #print ">>>>>>>>>>>>>>>>>>>>>>>>>setting skip for "+model+postmodel
                #                 skip = True
                #                 current_model = model+postmodel
                #             else:
                #                 current_model = model
                #         else:
                #             current_model = model               

                #     # state 0 means a new model:
                #     elif (state == '0') and skip:
                #         current_postmodel = postmodel
                #         skip = False


                #     # State 2 means we're at the end of the model
                #     # so let's save the previous one:

                #     elif (state == '2'):
                #         #print "In state 2"
                #         if not skip:
                #             current_end = end
                #             if debug:
                #                 print "pushing to array %s-%s %s-%s+%s" % (current_start, current_end, current_premodel, current_model, current_postmodel)
                #             if (current_model != '__'):
                #                 new_align.append({'pre' : current_premodel,
                #                                   'model' : current_model,
                #                                   'post' : current_postmodel,
                #                                   'start' : current_start,
                #                                   'end' : current_end,
                #                                   'triphone': str(current_premodel)+'-'+current_model+'+'+str(current_postmodel),
                #                                   'sortable': current_model+'--'+str(current_premodel)+'++'+str(current_postmodel)
                #                               })

                            
                #             labelstring += "." +current_model
                #             if skipmark:
                #                 labelstring += ". "
                #                 skipmark=False
                #             #current_premodel = current_model
                #         else:
                #             print "skipping "+premodel+"-"+model + "+"+postmodel+"?"


                # if lcounter > 0:       
                #     olderl = oldl
                # oldl = l

                # no_skipping = False
                # lcounter += 1

            phone_merge_handle.write("%s\t%s\n" % (labelfile, labelstring))

            # OK, label file done.
            # Now it's time to process the audio.
            # We'll send to the feature extractor the bits of the file that 
            # match the speech segments.

            #with open(audiofile, "rb") as binary_file:
            #f (1 == 1):

            # Read the whole file at once
            #data = binary_file.read()

            data = np.fromfile( audiofile, 'int16', -1)

            startmark = int(new_align[0]['start'])
            endmark= int(new_align[-1]['end'])
                
            if debug:
                print "start feature extraction at %s (%f s) and end at %s (%f s) ==> %i frames"  % (startmark, (float(startmark)/16000), endmark, (float(endmark)/16000), (endmark-startmark)/frame_step)


            # Communication from: 
            # http://stackoverflow.com/questions/163542/python-how-do-i-pass-a-string-into-subprocess-popen-using-the-stdin-argument

            #inputdata=data[startmark*datatypelength : (endmark + frame_leftovers)*datatypelength]
            #print inputdata

            tmp_input=os.path.join(tmp_dir,str(tmpfilecounter)+"_in")
            tmp_output=os.path.join(tmp_dir,str(tmpfilecounter)+"_out")

            data.tofile(tmp_input, "")

            #myfmt='f'*len(inputdata)
            ##  You can use 'd' for double and < or > to force endinness
            #bindata=struct.pack(myfmt,*inputdata)

            #tmp_features = Popen([feature_extraction_script, tmp_input, tmp_output ], shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate(input=bindata)
            #print(tmp_features)


            process_progress = Popen([feature_extraction_script, tmp_input, tmp_output, str(startmark), str(endmark+frame_leftovers) ], stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()

            feature_list = np.fromfile(tmp_output, dtype='float32', count=-1)
            feature_array = feature_list.reshape([-1,feature_dimension])

            f_end =  (int(new_align[-1]['end'])-startmark)/frame_step

            if debug:
                print "Utterance data size: %i x %i" % (feature_array).shape

            if (feature_array.shape[0] < f_end):
                    print "Not enough features for file %s: %i < %i" % (audiofile, feature_array.shape[0], f_end)
                    print "panic save to /tmp/this_is_not_good"
                    np.savetxt('/tmp/this_is_not_good', feature_array, delimiter='\t')
                    sys.exit(0)

            else:

                for l in new_align:                
                    lkey = l['sortable']
                    mkey = l['model']
                    tp = l['triphone']
                    all_trips_counter += 1

                    l_start = (int(l['start'])-startmark)/frame_step
                    l_end =  (int(l['end'])-startmark)/frame_step
                    l_length = l_end - l_start

                    if (feature_array.shape[0] < l_end):
                        print "Not enough features: %i < %i" % (feature_array.shape[0], l_end)

                    statistics_handle.write("%i\t%s\n" % (l_length, tp))

                    if debug:
                        print "---------------------------"
                        print "Array stats: start %i -> %i length ?? -> %i end %i -> %i"% (int(l['start'])-startmark, l_start, l_length, int(l['end'])-startmark, l_end )
                        print "      phone data size: %i x %i" % (feature_array[l_start:l_end, :]).shape


                    # If phone is shorter than maximum length, we'll pad with zeros:
                    if (l_length < max_num_frames):
                        if debug:
                            print " padding with "+ str(max_num_frames-l_length) +" zero vectors"
                        l_array = np.pad(feature_array[l_start:l_end, :], 
                                         ([0,max_num_frames-l_length],[0,0]),
                                         mode='constant', constant_values=0)

                    # If it's not shorter:
                    else:
                        # Perfect match with maximum length!!
                        if (l_length == max_num_frames):
                            l_array = feature_array[l_start:l_end, :]
   
                        # If we have a utterance start model, let's take the final bit from it:                            
                        elif l['pre']== '_':
                            l_array = feature_array[l_end-max_num_frames:l_end, :]
                        # If we have a utterance end model, let's take the first bit from it:
                        elif ['post'] == '_' :                            
                            l_array = feature_array[l_start:l_start+max_num_frames, :]

                        # same thing if we have a middle of the utterance model:
                        else:
                            l_array = feature_array[l_start:l_start+max_num_frames, :]

                        if (l_length > max_num_frames):
                            too_long_counter += 1
                            sys.stderr.write("\r%0.2f%s What a trouble! Triphone %s is too long (%i frames)! %i too long things already, that's %0.2f %s of all!\n" % (100.0*recipefilecounter/collection['numlines'], "%",tp, l_length, too_long_counter, 100.0*too_long_counter/all_trips_counter, "%"))

                    if debug:
                        print "Data size: %i x %i" % l_array.shape


                    if mkey not in triphonedata.keys():
                        triphonedata[mkey] = {}

                    if lkey not in triphonedata[mkey].keys():                    
                        triphonedata[mkey][lkey] = { 'data': np.zeros([assigned_num_samples, 
                                                                       max_num_frames, 
                                                                       feature_dimension], dtype='float32'),
                                                     'counter': 0,
                                                     'mono' :l['model'],
                                                     'triphone' : l['triphone']}
                        
                    tpc=triphonedata[mkey][lkey]['counter']


                    if (triphonedata[mkey][lkey]['data']).shape[0] <= tpc:
                        if debug:
                            print str((triphonedata[mkey][lkey]['data']).shape[0]) +" =< "+str(tpc) +"?"
                            print "Expand this triphone now:"
                            print triphonedata[mkey][lkey]['data'].shape
                        triphonedata[mkey][lkey]['data'] = np.append(triphonedata[mkey][lkey]['data'],
                                  np.zeros([assigned_num_samples, 
                                            max_num_frames, 
                                            feature_dimension], dtype='float32'),
                                  0)
                        if debug:
                            print triphonedata[mkey][lkey]['data'].shape
                    

                    triphonedata[mkey][lkey]['data'][tpc,:,:] = l_array
                    
                    triphonedata[mkey][lkey]['counter']+=1

                os.remove(tmp_input)
                os.remove(tmp_output)
                

        recipefilecounter += 1

        if recipefilecounter % file_batch_size == 0:
            # So much data we'll have to save to intermediate files:
            dummy = 1


        if not debug:
            if (recipefilecounter % int(progress_interval)) == 0:
                sys.stderr.write("\r%0.2f%s %s %s" % (100.0*recipefilecounter/collection['numlines'], "%",collection['condition'], collection['name'] ))
                sys.stderr.flush()

        if (recipefilecounter == collection['numlines']):
            break
        
    print "\n%s\n"%triphonedata.keys()
    
    for mono in sorted(triphonedata.keys()):
    
        grande_features =  np.zeros([0, max_num_frames, feature_dimension], dtype='float')
        grande_classes = np.zeros([0, max_num_classes ], dtype='float')

        for tripkey in sorted(triphonedata[mono].keys()):
        
            tripdata = triphonedata[mono][tripkey]
            trip =  tripdata['triphone']

            if trip in triphoneclasses.keys():
                tripcl = triphoneclasses[trip]
            elif (collection['name'][0:8] == 'training') or (collection['name'] == 'python-devel-training' ):
                tripcl = len(triphoneclasses)
                triphoneclasses[trip] = tripcl                
                class_handle.write("%s\t%i\t%i\n" % (trip, tripcl, tripdata['counter']))
            else:                
                continue
                
            grande_features = np.append(grande_features, tripdata['data'][0:tripdata['counter'],:,:], 0)

            piccolo_classes =  np.zeros([ max_num_classes ], dtype='float')
            piccolo_classes[tripcl] = 1
            grande_classes = np.append(grande_classes, np.tile(piccolo_classes,(tripdata['counter'],1)),0)

            

        modeldir_unicode = mono

        new_path=os.path.join(pickle_dir, collection['condition']+"-"+collection['name'])

        picklefile = os.path.join(new_path,  collection['condition']+"-"+collection['name']+"."+modeldir_unicode+".pkl")


        try:
            os.makedirs(new_path)

        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(new_path):
                pass
            else:
                raise   
            
        print "pickling %i items to %s" % ( grande_features.shape[0], picklefile);
                
        outf = open(picklefile, 'wb')
        
        # Pickle the list using the highest protocol available.
        cPickle.dump({'data': grande_features, 'classes': grande_classes}, outf, protocol=cPickle.HIGHEST_PROTOCOL)
                
