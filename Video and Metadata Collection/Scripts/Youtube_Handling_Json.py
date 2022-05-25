import json
import pandas as pd
import os

dict_list = []
with open('complete_data.txt', 'r') as json_file:
	dict_list = json_file.readlines()

# Deal with last line
lst = []

for i in dict_list:
	if dict_list.index(i) != len(dict_list) - 1:
		temp_vid_info = json.loads(i)
	else:
		temp_vid_info = json.loads(json.dumps(i))
	print(dict_list.index(i))
	print(temp_vid_info)
	# comments to be retrieved later
	if "comments" in temp_vid_info.keys():
		del temp_vid_info["comments"]
	lst.append(pd.json_normalize(temp_vid_info))

all_data_df = pd.concat(lst, axis=0).fillna(0)
# removing duplicates
all_data_df = all_data_df.drop_duplicates(subset='id', keep='first')
all_data_df.to_csv('metadata_unlabeled_videos.csv', index=False)