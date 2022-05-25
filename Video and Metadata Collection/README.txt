************************************************************************Pipeline Steps********************************************************************************* 

1. Run Script named: "Youtube_GoogleTrends.py". It takes as input an excel file named: "InputList.xlsx", containing 238 keywords and outputs a csv file 
   named: "NewList.csv" with two columns: 1) Filtered Keyword, 2) Average Popularity. The script used pytrends to create a comparison of 238 keywords to produce a 
   filtered file with 100 most popular keywords. It also creates a folder named: "googletrends", which is used for processing, but is not important for the pipeline.

2. Run Script named: "Youtube_Search.py". It takes as input "NewList.csv" file and returns a text file named: "videoIDs.txt", which contains the videoIDs for 
   search results from the keywords. This script also returns a text file named: "complete_data.txt", which contains the raw metadata for these videos in json format. 
   Note that the script named: "Youtube_Metadata.py", which does the webscraping using Youtube_API and Sellenium (chrome driver installation required), has pip already been imported into this script.

3. Run Script named: "Youtube_Downloader.py". It takes as input "videoIDs.txt" from step 2 and downloads all those videos to a folder named: "videos" in the 
   current directory of the local machine.

4. Run Script named: "Youtube_Upload2Cloud.py". It takes as input a folder named "videos", and uploads all video files in that folder to a specified bucket
   on GCP. This script requires Google License and Project Ownership on GCP to avoid permission issues in uploading.

5. Run Script named: "Youtube_Handling_Json.py". It takes as input "complete_data.txt" from step 2 and returns a csv file named: "metadata_unlabeled_videos.csv", which
   contains all metadata features except comments.

6. Run Script named: "Youtube_Features_Unlabeled_Videos.py". It takes as input "complete_data.txt" from step 2 and returns a csv file named: "features_unlabeled_videos
   .csv". This file contains the metadata view features (except MER) that will be used for Machine Learning models.

7. Run Script named: "Youtube_Features_Labeled_Videos.py". It takes as input "metadata_labeled_videos.csv" already provided by Xiao, and returns a csv file 
   named: "features_labeled_videos.csv". This file contains the metadata view features that will be used for Machine Learning models.



**********************************************************************Input/Output Summary***************************************************************************** 

The INPUT required for this process is an excel file named: 
1) "InputList.xlsx"; 
2) "metadata_labeled_videos.csv" since this was already collected from prior work. 

The OUTPUT from this process would be (1, 4, 5 not uploaded to GitHub due to file size issues): 
1) "googletrends" folder containing csv files only used for some processing;
2) "NewList.csv" containing 100 filtered keywords and their average popularity; 
3) "videoIDs.txt" containing the video IDs of results from youtube search using filtered keywords; 
4) "complete_data.txt" containing the metadata for all unlabeled videos; 
5) "videos" folder containing downloaded videos; 
6) "metadata_unlabeled_videos.csv" containing the raw metadata; 
7) "features_unlabeled_videos.csv" containing the features extracted from the raw metadata of unlabeled videos; 
8) "features_labeled_videos.csv" containing the features extracted from the raw metadata of Xiao's labeled videos.

Note: Additional details about each Python script have been provided as comments within the respective scripts.
