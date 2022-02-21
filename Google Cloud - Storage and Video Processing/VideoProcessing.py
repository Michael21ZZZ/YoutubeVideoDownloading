#get_ipython().system('pip install --upgrade google-cloud-videointelligence')

import os, io
import pandas as pd
from google.cloud import videointelligence
from collections import namedtuple
import plotly.graph_objects as go
import plotly.express as px
import requests
import string
import proto
import json

## download your API Key from your Google Console

#this is the path to the API credentials. Modify this path to the json file in this folder.
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'stoked-dominion-341317-faec5815b985.json'
video_client = videointelligence.VideoIntelligenceServiceClient()


import difflib

# Changes:
# swapped positions of cat and dog
# changed gophers to gopher
# removed hound
# added mouse

for line in difflib.unified_diff(lines1, lines2, fromfile='file1', tofile='file2', lineterm='', n=0):
    print(line)


# In[ ]:


def compare(File1,File2):
    with open(File1,'r') as f:
        d=set(f.readlines())


    with open(File2,'r') as f:
        e=set(f.readlines())

    open('file3.txt','w').close() #Create the file

    with open('file3.txt','a') as f:
        for line in list(d-e):
           f.write(line)



from google.cloud import storage

output_list_normal = []
def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):

    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    print("Blobs:")
    for blob in blobs:
      print(blob.name)
      output_list_normal.append(blob.name)


from google.cloud import storage

video_list = []
def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):

    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    print("Blobs:")
    for blob in blobs:
      print(blob.name)
      video_list.append(blob.name)


# In[ ]:


if __name__ == "__main__":
    list_blobs_with_prefix(
        bucket_name="video_list_capstoneyt_2022", prefix="outputOCR/", delimiter="/"
    )



# this is a video that I have uploaded on Cloud Storage and I obtained the URI from google console
for video in video_list:
  print(video_list.index(video))
  gs_URI = 'gs://youtube_videos_data/'+ video
  temp = video.split('/')
  print(temp)
  print("here", gs_URI)
  #output_uri = gs_URI + '.json'
  features = [videointelligence.Feature.SPEECH_TRANSCRIPTION, videointelligence.Feature.SHOT_CHANGE_DETECTION, videointelligence.Feature.OBJECT_TRACKING, videointelligence.Feature.TEXT_DETECTION]
  config = videointelligence.SpeechTranscriptionConfig(language_code="en-US", enable_automatic_punctuation=True, enable_word_confidence=False)
  config1 = videointelligence.ShotChangeDetectionConfig()
  config2 = videointelligence.ObjectTrackingConfig()
  config3 = videointelligence.TextDetectionConfig()
  video_context = videointelligence.VideoContext(speech_transcription_config=config, shot_change_detection_config=config1, object_tracking_config=config2,text_detection_config=config3)
  operation = video_client.annotate_video(
    request={
        "features": features,
        "input_uri": gs_URI,
        "video_context": video_context,
        "output_uri": 'gs://youtube_videos_data/output/'+ temp[1] +'.json'
    }
  )

  print("\nProcessing video for speech transcription.")

  result = operation.result(timeout=100000000)

# There is only one annotation_result since only
# one video is processed.
  annotation_results = result.annotation_results[0]
  print("Completed")

# this is a video that I have uploaded on Cloud Storage and I obtained the URI from google console
print("out")
for video in video_list:
  print("in")
  print(video_list.index(video))
  gs_URI = 'gs://youtube_videos_data/'+ video
  temp = video.split('/')
  print(temp)
  print("here", gs_URI)
  features = [videointelligence.Feature.TEXT_DETECTION]
  config = videointelligence.TextDetectionConfig()
  video_context = videointelligence.VideoContext(text_detection_config=config)
  operation = video_client.annotate_video(
    request={
        "features": features,
        "input_uri": gs_URI,
        "video_context": video_context,
        "output_uri": 'gs://youtube_videos_data/outputOCR/'+ temp[1] +'OCR'+'.json'
    }
  )

  print("\nProcessing video for speech transcription.")

  result = operation.result(timeout=100000000)

# There is only one annotation_result since only
# one video is processed.
  annotation_results = result.annotation_results[0]
  print("Completed")
