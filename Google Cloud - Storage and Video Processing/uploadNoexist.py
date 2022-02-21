from google.cloud import storage
import os
import argparse

"""
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


def get_exist_filename_set(bucket_name):

    client = storage.Client()
    # bucket = storage.Bucket("videos")
    blob_list = client.list_blobs(bucket_name)
    filenames = set()
    for blob in blob_list:
        filenames.add(blob.name)
        # print(blob.name)
    return filenames


def upload_NotExist_blobs(bucket_name):
    filenames = get_exist_filename_set(bucket_name)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    file_names = os.listdir(UP_LOAD_FILE_ROUTE)
    for file1 in file_names:
        if file1 in filenames:
            print("File {} exists.".format(file1))
            continue
        source_file_route = os.path.join(UP_LOAD_FILE_ROUTE, file1)
        blob = bucket.blob(file1)
        blob.upload_from_filename(source_file_route)
        print(
            "File {} uploaded to {}.".format(
                source_file_route, file1
            )
        )


if __name__ == "__main__":
    upload_NotExist_blobs(BUCKET_NAME)
