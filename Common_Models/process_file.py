#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:46:25 2023

@author: wangkai
"""

from pydub import AudioSegment
from pydub.utils import make_chunks
import os, re
import json
import pathlib

#wav_url = "Good_AV/"
#save_wav = "Prepared_dataset/good/"

wav_url = "Bad_AV/"
save_wav = "Prepared_dataset/bad/"

in_path = pathlib.Path.cwd().joinpath(wav_url)
out_path = pathlib.Path.cwd().joinpath(save_wav)

for each in os.listdir(wav_url):
    filename = re.findall(r"(.*?)\.mp3",each)
    print (each)
    path = in_path.joinpath(each)
    print (path)
    if (each.endswith(".m4a")):
        print(each)
        print("m4a type")
        mp3 = AudioSegment.from_file(path,"m4a")
    elif (each.endswith(".mp3")):
        print(each)
        print ("mp3 type")
        mp3 = AudioSegment.from_file(path,"mp3")
    elif (each.endswith(".wav")):
        print(each)
        print ("wav type")
        mp3 = AudioSegment.from_file(path,"wav")
    elif (each.endswith(".aac")):
        print(each)
        print ("aac type")
        mp3 = AudioSegment.from_file(path,"aac")
    else:
        next
    
    print(each)
    size = 10000
    chunks = make_chunks(mp3, size)
        
    for i, chunk in enumerate(chunks):
        chunk_name = "{}-{}.mp3".format(each.split(".")[0], i)
        print(chunk_name)
        o_path = out_path.joinpath(chunk_name)
        chunk.export(o_path, format="mp3")
