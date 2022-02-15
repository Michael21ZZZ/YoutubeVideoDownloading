# Imports the Google Cloud client library
from google.cloud import storage
import os
import argparse
import pprint


"""
#uncomment if using arguments in console
parser = argparse.ArgumentParser()
parser.add_argument('--audio_path', type=str, default='./audios')
parser.add_argument('--bucket_name', type=str, default='')
args = parser.parse_args()

UP_LOAD_FILE_ROUTE = args.audio_path
BUCKET_NAME = args.bucket_name
"""

#path of the files
UP_LOAD_FILE_ROUTE = "../Toy_data"

#the name of the bucket
BUCKET_NAME ="video_list_capstoneyt_2022"


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    #bucket_name = "video_list_capstoneyt_2022"
    #source_file_name = "../Toy_data/The_Greatest_Show.mp4"
    #destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
# upload_blob('videos', './test.wav', 'test.wav')


def bucket_metadata(bucket_name):
    """Prints out a bucket's metadata."""
    # bucket_name = 'your-bucket-name'

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    print("ID: {}".format(bucket.id))
    print("Name: {}".format(bucket.name))
    print("Storage Class: {}".format(bucket.storage_class))
    print("Location: {}".format(bucket.location))
    print("Location Type: {}".format(bucket.location_type))
    print("Cors: {}".format(bucket.cors))
    print(
        "Default Event Based Hold: {}".format(bucket.default_event_based_hold)
    )
    print("Default KMS Key Name: {}".format(bucket.default_kms_key_name))
    print("Metageneration: {}".format(bucket.metageneration))
    print(
        "Retention Effective Time: {}".format(
            bucket.retention_policy_effective_time
        )
    )
    print("Retention Period: {}".format(bucket.retention_period))
    print("Retention Policy Locked: {}".format(bucket.retention_policy_locked))
    print("Requester Pays: {}".format(bucket.requester_pays))
    print("Self Link: {}".format(bucket.self_link))
    print("Time Created: {}".format(bucket.time_created))
    print("Versioning Enabled: {}".format(bucket.versioning_enabled))
    print("Labels:")
    pprint.pprint(bucket.labels)


def upload_files(bucket_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    file_names = os.listdir(UP_LOAD_FILE_ROUTE)
    for file1 in file_names:
        source_file_route = os.path.join(UP_LOAD_FILE_ROUTE, file1)
        blob = bucket.blob(file1)
        blob.upload_from_filename(source_file_route)
        print(
            "File {} uploaded to {}.".format(
                source_file_route, file1
            )
        )

upload_files(BUCKET_NAME)
