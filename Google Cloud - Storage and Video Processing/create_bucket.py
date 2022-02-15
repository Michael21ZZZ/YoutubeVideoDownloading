from google.cloud import storage
import os
import argparse
"""
parser = argparse.ArgumentParser()
parser.add_argument('--bucket_name', type=str, default='')
args = parser.parse_args()

bucket_name = args.bucket_name
"""
# Instantiates a client
storage_client = storage.Client()

# The name for the new bucket
bucket_name = "video_list_capstoneyt_2022"
print(bucket_name)

# Creates the new bucket
bucket = storage_client.create_bucket(bucket_name)

print("Bucket {} created.".format(bucket.name))
