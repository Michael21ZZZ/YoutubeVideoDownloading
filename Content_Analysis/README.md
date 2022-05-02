From a generalized perspective, our feature extraction process requires the following scripts(for more specifics details, refer to Diagram 1):

1. Script to download Youtube Video files to Google Drive
2. Script for video to audio extraction on Google Drive
3. Script to upload files from Google Drive to GCP
4. Script to process features
   a. video metadata feature extraction (.mp4 to .csv and .txt)
   b. video content analysis via Google Cloud Video Intelligence Platform ( .mp4 to .json)
   c. scripts to process audio transcription  (.mp4 to .wav to .txt) 
   d. script to process speaker feature (gender) (.wav to .txt)
   e. script for processing MER features for video description (video metadata) and video transcription (video content) (.txt. to .txt)
5. Script to extract features from 4.b and 4.d that is stored on GCP
6. Script to concatenate features with MER features from 4.e (final result: one csv for unlabeled)


