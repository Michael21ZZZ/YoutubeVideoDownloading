from google.cloud import storage
from google.cloud import speech
#from google.cloud.speech_v1 import enums
#from google.cloud.speech_v1 import types
import io
import os
import argparse
import json

"""
parser = argparse.ArgumentParser()
# parser.add_argument('--audio_path', type=str, default='./audios')
parser.add_argument('--bucket_name', type=str, default='fyanvideos')
parser.add_argument('--save_path', type=str, default='./texts')
args = parser.parse_args()

# UP_LOAD_FILE_ROUTE = args.audio_path
BUCKET_NAME = args.bucket_name
SAVE_PATH = args.save_path
"""
# UP_LOAD_FILE_ROUTE = args.audio_path
SAVE_PATH = 'text'

#the name of the bucket
BUCKET_NAME ="video_list_capstoneyt_2022"


def sample_long_running_recognize(storage_uri, filename):
    """
    Transcribe long audio file from Cloud Storage using asynchronous speech
    recognition
    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    client = speech.SpeechClient()

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000

    # The language of the supplied audio
    language_code = "en-US"

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "audio_channel_count": 2,
        "enable_separate_recognition_per_channel": False,
        # "sample_rate_hertz": sample_rate_hertz,
        "language_code": language_code,
        "enable_automatic_punctuation": True,
        "encoding": encoding,
    }
    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()

    filename = filename.split('.')[0]

    transcript_json = []
    transcript_str = ''
    for idx, result in enumerate(response.results):
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))
        print(u"Confidence: {}".format(alternative.confidence))
        transcript_str += alternative.transcript
        transcript_json.append({"transcript":alternative.transcript, "confidence":alternative.confidence})
    with open(os.path.join(SAVE_PATH, filename+'.txt'), 'w') as f1:
        f1.write(transcript_str)
    with open(os.path.join(SAVE_PATH, filename+'.json'), 'w') as f2:
        json.dump(transcript_json, f2)


def get_exist_filename_set(bucket_name):

    client = storage.Client()
    # bucket = storage.Bucket("fyanvideos")
    blob_list = client.list_blobs(bucket_name)
    filenames = set()
    for blob in blob_list:
        filenames.add(blob.name)
        # print(blob.name)
    return filenames


def get_processed_filename_set():
    processed_filename_set = os.listdir(SAVE_PATH)
    processed_filename_set = [i.split('.')[0] for i in processed_filename_set]
    return processed_filename_set


def main():
    filenames = get_exist_filename_set(BUCKET_NAME)
    processed_filename_set = get_processed_filename_set()
    for filename in filenames:
        pre_name = filename.split('.')[0]
        if pre_name not in processed_filename_set:
            storage_uri = 'gs://{}/{}'.format(BUCKET_NAME, filename)
            sample_long_running_recognize(storage_uri, filename)


if __name__ == '__main__':
    main()
