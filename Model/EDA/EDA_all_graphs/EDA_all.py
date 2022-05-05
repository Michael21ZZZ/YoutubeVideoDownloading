import pandas as pd
import numpy as np
import isodate
import matplotlib.pyplot as plt
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import string
import codecs
from scipy import stats

labeled_df = pd.read_csv('Data_Xiao_fixed.csv', index_col = 0)
labeled_df_UND_group_duration = labeled_df[['UND', 'contentSeconds']].groupby('UND').agg(
					  {'contentSeconds':['mean','std']})
labeled_df_UND_group_duration.reset_index(inplace=True)

def group_data(data, col1, col2):
	labeled_df_UND_group_duration = data[[col1, col2]].groupby(col2).agg(
					  {col1:['mean','std']})
	return(labeled_df_UND_group_duration)

def group_data_both(data, col1, col2, col3):
	labeled_df_UND_group_duration = data[[col1, col2, col3]].groupby([col2, col3]).agg(
					  {col1:['mean','std']})
	return(labeled_df_UND_group_duration)

def plot_avg_std_duration(column_name, data, title):
	x = np.array(['Low Understandability', 'High Understandability'])
	y = np.array(list(data[column_name]['mean']))
	e = np.array(list(data[column_name]['std']))
	# print(data)
	plt.errorbar(x, y, e, linestyle='None', marker='o', capsize=3)
	plt.title(title)
	plt.annotate('Mean = ' + str(round(data[column_name]['mean'][0],2)), (0.05,422.32), fontsize = 7)
	plt.annotate('Mean = ' + str(round(data[column_name]['mean'][1],2)), (0.80,550), fontsize = 7)
	plt.annotate('Std = ' + str(round(data[column_name]['std'][0],2)), (0.05,380.32), fontsize = 7)
	plt.annotate('Std = ' + str(round(data[column_name]['std'][1],2)), (0.80,502), fontsize = 7)

	plt.show()

def plot_avg_std_dislike_count(column_name, data, title):
	x = np.array(['Low Understandability', 'High Understandability'])
	y = np.array(list(data[column_name]['mean'])) # Effectively y = x**2
	e = np.array(list(data[column_name]['std']))

	plt.errorbar(x, y, e, linestyle='None', marker='o', capsize=3)
	plt.title(title)
	plt.annotate('Mean = ' + str(round(data[column_name]['mean'][0],2)), (0.05,291), fontsize = 7)
	plt.annotate('Mean = ' + str(round(data[column_name]['mean'][1],2)), (0.80,115), fontsize = 7)
	plt.annotate('Std = ' + str(round(data[column_name]['std'][0],2)), (0.05,201), fontsize = 7)
	plt.annotate('Std = ' + str(round(data[column_name]['std'][1],2)), (0.80,25), fontsize = 7)

	plt.show()

# grouped_data = group_data(labeled_df, 'contentSeconds', 'UND')
# print(grouped_data)
# plot_avg_std_duration('contentSeconds', grouped_data, 'Average Video Duration for Understandability')


def plot_best_channels(labeled_df):
	grouped_data = labeled_df[['channelTitle', 'UND']].groupby('channelTitle').agg(
					  {'UND':['sum', 'count']})

	grouped_data = grouped_data.reset_index()
	channel_titles = np.array(list(grouped_data['channelTitle']))
	sum_und = np.array(list(grouped_data['UND']['sum']))
	count_und = np.array(list(grouped_data['UND']['count']))

	perc_und = (sum_und / count_und)*100
	perc_count_und = zip(perc_und, count_und, channel_titles)
	high_conf_videos = []
	low_conf_videos = []
	for (perc, count, title) in perc_count_und:
		if(count >= 3 and perc >= 85):
			high_conf_videos.append((perc, count, title))
		elif(perc <= 50):
			low_conf_videos.append((perc, count, title))

	high_conf_videos = np.array(high_conf_videos)
	low_conf_videos = np.array(low_conf_videos)

	percentages = high_conf_videos[:, 0].astype('float')

	titles = high_conf_videos[:, 2]
	percentages_sorted_idx = np.argsort(-percentages)
	titles_sorted = titles[percentages_sorted_idx]
	fig = plt.figure()
	plt.bar(titles_sorted, percentages[percentages_sorted_idx])
	plt.xticks(rotation = 90)
	fig.subplots_adjust(bottom=0.8)
	plt.show()

grouped_data = group_data(labeled_df, 'contentSeconds', 'UND')
grouped_data = group_data_both(labeled_df, 'contentSeconds', 'UND', 'MED').reset_index()

# print(grouped_data)
labels = ['UND=0, MED=0', 'UND=0, MED=1', 'UND=1, MED=0', 'UND=1, MED=1']
grouped_data_cs = grouped_data['contentSeconds']
means = list(grouped_data_cs['mean'])
stds = np.array(list(grouped_data_cs['std'])) / 2

ax = plt.subplot(111)

ax.errorbar(labels, means, stds, linestyle='None', marker='o', capsize=3)
plt.title('Average length of a video (seconds) for each group')
plt.ylabel('length (seconds)')

plt.annotate('Mean = ' + str(round(means[0],2)), (0.10,582.2), fontsize = 7)
plt.annotate('Std = ' + str(round(stds[0],2)), (0.10,552.2), fontsize = 7)

plt.annotate('Mean = ' + str(round(means[1],2)), (1.1,186), fontsize = 7)
plt.annotate('Std = ' + str(round(stds[1],2)), (1.1,156), fontsize = 7)

plt.annotate('Mean = ' + str(round(means[2],2)), (2.1,590.32), fontsize = 7)
plt.annotate('Std = ' + str(round(stds[2],2)), (2.1,560.32), fontsize = 7)

plt.annotate('Mean = ' + str(round(means[3],2)), (3.1,556), fontsize = 7)
plt.annotate('Std = ' + str(round(stds[3],2)), (3.1,526), fontsize = 7)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.show()


grouped_data = group_data_both(labeled_df, 'viewCount', 'UND', 'MED').reset_index()

print(grouped_data)
labels = ['UND=0, MED=0', 'UND=0, MED=1', 'UND=1, MED=0', 'UND=1, MED=1']
grouped_data_cs = grouped_data['viewCount']
means = list(grouped_data_cs['mean'])
stds = np.array(list(grouped_data_cs['std'])) / 2

ax = plt.subplot(111)

ax.errorbar(labels, means, stds, linestyle='None', marker='o', capsize=3)
plt.title('Average number of views for a video in each group')
plt.ylabel('Number of Views')

plt.annotate('Mean = ' + "{:,}".format(round(means[0],0)), (0.10,972000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[0],0)), (0.10,832000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[1],0)), (1.1,52000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[1],0)), (1.1,-88000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[2],0)), (2.1,470000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[2],0)), (2.1,330000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[3],0)), (3.1,300000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[3],0)), (3.1,160000), fontsize = 7)

# plt.axis('off')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.show()

grouped_data = group_data_both(labeled_df, 'channelViewCount', 'UND', 'MED').reset_index()

print(grouped_data)
labels = ['UND=0, MED=0', 'UND=0, MED=1', 'UND=1, MED=0', 'UND=1, MED=1']
grouped_data_cs = grouped_data['channelViewCount']
means = list(grouped_data_cs['mean'])
stds = np.array(list(grouped_data_cs['std'])) / 2

ax = plt.subplot(111)

ax.errorbar(labels, means, stds, linestyle='None', marker='o', capsize=3)
plt.title('Average number of views for a channel in each group')
plt.ylabel('Number of Views')

plt.annotate('Mean = ' + "{:,}".format(round(means[0],0)), (0.10,392000000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[0],0)), (0.10,338000000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[1],0)), (1.1,52000000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[1],0)), (1.1,17000000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[2],0)), (2.1,275000000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[2],0)), (2.1,230000000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[3],0)), (3.05,215000000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[3],0)), (3.05,170000000), fontsize = 7)
plt.show()

grouped_data = group_data_both(labeled_df, 'channelSubscriberCount', 'UND', 'MED').reset_index()

print(grouped_data)
labels = ['UND=0, MED=0', 'UND=0, MED=1', 'UND=1, MED=0', 'UND=1, MED=1']
grouped_data_cs = grouped_data['channelSubscriberCount']
means = list(grouped_data_cs['mean'])
stds = np.array(list(grouped_data_cs['std'])) / 2

ax = plt.subplot(111)

ax.errorbar(labels, means, stds, linestyle='None', marker='o', capsize=3)
plt.title('Average number of subscribers for a channel in each group')
plt.ylabel('Number of Subscribers')

plt.annotate('Mean = ' + "{:,}".format(round(means[0],0)), (0.10,946000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[0],0)), (0.10,855000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[1],0)), (1.1,229000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[1],0)), (1.1,129000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[2],0)), (2.1,830000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[2],0)), (2.1,730000), fontsize = 7)

plt.annotate('Mean = ' + "{:,}".format(round(means[3],0)), (3.05,830000), fontsize = 7)
plt.annotate('Std = ' + "{:,}".format(round(stds[3],0)), (3.05,730000), fontsize = 7)

# plt.axis('off')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.show()

def year_wise_counts(labeled_df):
	df = labeled_df[['publishedAt', 'UND', 'MED']]
	dates = list(df['publishedAt'])
	years = np.array([y.split('-')[0] for y in dates])
	df['years'] = years
	df_grouped = df.groupby(['UND', 'MED', 'years'])
	grouped_count = df_grouped.count()
	idx = grouped_count.index
	counts = grouped_count.values
	hist_dict = {}
	total_videos = 0
	for (und, med, year), count in zip(idx, counts):
		hist_dict[(und, med, year)] = 0
	for (und, med, year), count in zip(idx, counts):
		hist_dict[(und, med, year)] += count[0]
		total_videos += count[0]
	
	# print(hist_dict)
	
	year_00 = {}
	year_01 = {}
	year_10 = {}
	year_11 = {}

	year_wise_counts_dict = {}

	for (und, med, year), count in hist_dict.items():
		if(und == 0 and med == 0):
			year_00[year] = count
		if(und == 0 and med == 1):
			year_01[year] = count
		if(und == 1 and med == 0):
			year_10[year] = count
		if(und == 1 and med == 1):
			year_11[year] = count
		
		if(year not in year_wise_counts_dict):
			year_wise_counts_dict[year] = 0
		else:
			year_wise_counts_dict[year] += count

	# print(year_und)
	# print(year_med)
	print(year_wise_counts_dict)

	list_00 = []
	list_01 = []
	list_10 = []
	list_11 = []

	ordered_year_list = sorted([int(i) for i in list(set(years))])
	ordered_year_list_str = [str(i) for i in ordered_year_list]
	# print(ordered_year_list)
	
	for year in ordered_year_list_str:
		if(year in year_00):
			list_00.append(year_00[year])
		else:
			list_00.append(0)

		if(year in year_01):
			list_01.append(year_01[year])
		else:
			list_01.append(0)

		if(year in year_10):
			list_10.append(year_10[year])
		else:
			list_10.append(0)

		if(year in year_11):
			list_11.append(year_11[year])
		else:
			list_11.append(0)

	# print(len(list_00))
	# print(len(list_01))
	# print(len(list_10))
	# print(len(list_11))
	# print(len(years))
	# print(year_11)

	labels = ordered_year_list
	# print(year_wise_counts_dict)
	year_counts = []
	for y in labels:
		year_counts.append(year_wise_counts_dict[str(y)])


	print(year_counts)
	new_df = pd.DataFrame(zip(labels, list_00, list_01, list_10, list_11), columns = ['Year',
																		'UND=0, MED=0',
																		'UND=0, MED=1',
																		'UND=1, MED=0',
																		'UND=1, MED=1'])

	# new_df = pd.DataFrame(zip(labels, list_00, list_01, list_10, list_11), columns = ['Year',
	# 																	'UND=0, MED=0',
	# 																	'UND=0, MED=1',
	# 																	'UND=1, MED=0',
	# 																	'UND=1, MED=1'])
	
	
	# new_df['Count'] = year_counts
	print(new_df)
	und_df = pd.DataFrame()
	und_df['Year'] = new_df['Year']
	# und_df['Count'] = new_df['Count']
	und_df['UND=0'] = new_df['UND=0, MED=0'] + new_df['UND=0, MED=1']
	und_df['UND=1'] = new_df['UND=1, MED=0'] + new_df['UND=1, MED=1']
	und_df['Total'] = und_df['UND=0'] + und_df['UND=1']
	# und_df['UND=0'] = und_df['UND=0'] / und_df['Total']
	# und_df['UND=1'] = und_df['UND=1'] / und_df['Total']
	und_df = und_df.drop(columns = ['Total'])
	med_df = pd.DataFrame()
	med_df['Year'] = new_df['Year']
	# med_df['Count'] = new_df['Count']
	med_df['MED=0'] = new_df['UND=0, MED=0'] + new_df['UND=1, MED=0']
	med_df['MED=1'] = new_df['UND=0, MED=1'] + new_df['UND=1, MED=1']
	med_df['Total'] = med_df['MED=0'] + med_df['MED=1']
	# med_df['MED=0'] = med_df['MED=0'] / med_df['Total']
	# med_df['MED=1'] = med_df['MED=1'] / med_df['Total']
	med_df = med_df.drop(columns = ['Total'])
	new_df = new_df.drop(columns = ['UND=0, MED=0', 'UND=0, MED=1', 'UND=1, MED=0', 'UND=1, MED=1'])

	# print(und_df)
	# print(und_df)
	# print(counts)
	ax = plt.subplot(111)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	med_df.plot(
	    x = 'Year',
	    kind = 'bar',
	    stacked = False,
	    title = 'Count of videos having high / low medical information',
	    color = ['darkgrey', 'goldenrod'],
    	mark_right = True)
	plt.show()

	und_df.plot(
	    x = 'Year',
	    kind = 'bar',
	    stacked = False,
	    title = 'Count of videos having high / low understandability',
	    color = ['darkgrey', 'goldenrod'],
    	mark_right = True)
	plt.show()
	# men_means = [20, 35, 30, 35, 27]
	# women_means = [25, 32, 34, 20, 25]
	# men_std = [2, 3, 4, 1, 2]
	# women_std = [3, 5, 2, 3, 3]
	width = 0.35       # the width of the bars: can also be len(x) sequence

	fig, ax = plt.subplots()

	ax.bar(labels, list_00, width, label='UND = 0, MED = 0')
	ax.bar(labels, list_01, width, bottom=list_00, label='UND = 0, MED = 1')
	# ax.bar(labels, list_10, width, bottom=list_01, label='UND = 1, MED = 0')
	# ax.bar(labels, list_11, width, bottom=list_10, label='UND = 1, MED = 1')

	ax.set_ylabel('Counts')
	ax.set_xlabel('Years')
	ax.set_title('Distribution of videos across years')
	ax.legend()

	# plt.show()


	y_vals, counts = np.unique(years, return_counts = True)
	# print(years)
	# print(y_vals)
	# print(counts)
	# plt.bar(y_vals, counts)
	# plt.show()

def corr_und(labeled_df):
	# print(labeled_df)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labels = ['dislikeCount', 'channelSubscriberCount', 'channelViewCount',
							'commentCount', 'viewCount', 'channelCommentCount',
							'channelVideoCount', 'contentSeconds', 'likeCount']

	filtered = labeled_df[['dislikeCount', 'channelSubscriberCount', 'channelViewCount',
							'commentCount', 'viewCount', 'channelCommentCount',
							'channelVideoCount', 'contentSeconds', 'likeCount']]
	corr_mat = filtered.corr()
	# # corr_mat.style.background_gradient(cmap='hot')
	aaa = ax.matshow(corr_mat, cmap='hot')
	fig.colorbar(aaa)
	xaxis = np.arange(len(labels))
	plt.xticks(rotation = 90)
	# ax.set_cmap('hot')
	ax.set_xticks(xaxis)
	ax.set_yticks(xaxis)
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	plt.show()
	# und = list(filtered['MED'])
	# corrs = {}
	# for col in labels:
	# 	list_col = filtered[col]
	# 	corr = stats.pointbiserialr(und, list_col).correlation
	# 	corrs[col] = corr

	# for k, v in corrs.items():
	# 	print(k, ':', v)
	# a = list(filtered['dislikeCount'])
	
	# corr = stats.pointbiserialr(a, b).correlation
	# print(corr)

corr_und(labeled_df)

def med_counts(labeled_df):
	labeled_df_filtered = labeled_df[['UND', 'MED', 'URL']].groupby(['MED', 'UND']).count().reset_index()
	# labeled_df_filtered.plot(
	#     x = 'MED',
	#     kind = 'bar',
	#     stacked = False,
	#     title = 'Count of videos having high / low understandability',
	#     color = ['darkgrey', 'goldenrod'],
 #    	mark_right = True)
	# plt.show()
	N = 2
	print(labeled_df_filtered)
	counts = list(labeled_df_filtered['URL'])
	print(counts)

	blue_bar = (counts[0], counts[2])
	# Specify the values of orange bars (height)
	orange_bar = (counts[1], counts[3])

	# Position of bars on x-axis
	ind = np.arange(N)

	# Figure size
	plt.figure(figsize=(10,5))

	# Width of a bar 
	width = 0.3       

	# Plotting
	ax = plt.subplot(111)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.bar(ind, blue_bar , width, label='UND=0')
	ax.bar(ind + width, orange_bar, width, label='UND=1')

	plt.xlabel('MED')
	plt.ylabel('Number of Videos')
	plt.title('Comparison of the number of highly understandable videos in each category of Medical Information')

	# xticks()
	# First argument - A list of positions at which ticks should be placed
	# Second argument -  A list of labels to place at the given locations
	plt.xticks(ind + width / 2, ('0', '1'))

	# Finding the best position for legends and putting it
	plt.legend(loc='best')

	plt.annotate(str(counts[0]), (0,95), fontsize = 7)
	plt.annotate(str(counts[1]), (0.25,139), fontsize = 7)
	plt.annotate(str(counts[2]), (1,70), fontsize = 7)
	plt.annotate(str(counts[3]), (1.255,327), fontsize = 7)
	# plt.annotate('Mean = ' + str(round(data[column_name]['mean'][1],2)), (0.80,550), fontsize = 7)
	# plt.annotate('Std = ' + str(round(data[column_name]['std'][0],2)), (0.05,380.32), fontsize = 7)
	# plt.annotate('Std = ' + str(round(data[column_name]['std'][1],2)), (0.80,502), fontsize = 7)

	plt.show()

	# labeled_df_filtered_med_0 = labeled_df_filtered[labeled_df_filtered['MED'] == 0]
	# labeled_df_filtered_med_1 = labeled_df_filtered[labeled_df_filtered['MED'] == 1]

	# print(labeled_df_filtered_med_1)

def make_pie_chart(labeled_df):
	labeled_df_filtered = labeled_df[['UND', 'MED', 'URL']].groupby(['MED', 'UND']).count().reset_index()
	print(labeled_df_filtered)
	counts = list(labeled_df_filtered['URL'])
	labels = ['MED=0, UND=0', 'MED=0, UND=1', 'MED=1, UND=0', 'MED=1, UND=1']
	
	colors = ['#FF0000', '#0000FF', '#FFFF00', 
	          '#ADFF2F', '#FFA500']
	explode = (0.05, 0.05, 0.05, 0.05)
	plt.pie(counts, colors=colors, labels=labels,
	        autopct='%1.1f%%', pctdistance=0.85,
	        explode=explode)
	centre_circle = plt.Circle((0, 0), 0.70, fc='white')
	fig = plt.gcf()
	fig.gca().add_artist(centre_circle)
	plt.title('Percentage of videos in each category')
	plt.show()

import seaborn as sns
from scipy.stats import gaussian_kde

def make_density_plots(labeled_df):
	filtered = labeled_df[['UND', 'likeCount']]
	filtered_0 = filtered[filtered['UND'] == 0]
	filtered_1 = labeled_df[labeled_df['UND'] == 1]
	lst_0 = list(filtered_0['likeCount'])
	# print(filtered_0)
	
	# print(lst_0)
	plt.hist(lst_0, bins=60)
	plt.show()
	lst_1 = list(filtered_1['likeCount'])
	lst_0 = list(filtered_0['likeCount'])
	sns.kdeplot(lst_0, color="green", shade=True, label = 'Low Understandability')
	sns.kdeplot(lst_1, color="blue", shade=True, label = 'High Understandability')
	plt.xlim(-20000, 20000)
	plt.title('Density plot for the number of likes for a video')
	plt.legend(loc='best')
	plt.show()


	filtered_0['likeCount'].plot(kind='density', label='Low Understandability')
	plt.show()
	# print(filtered_1)
	filtered_1['viewCount'].plot(kind='density', label = 'High Understandability')
	plt.legend(loc='best')
	
	plt.show()

def channel_wise_count_und(labeled_df):
	filtered = labeled_df[['channelTitle', 'UND', 'URL']]
	filtered_grouped = filtered.groupby(['channelTitle', 'UND']).count().reset_index().sort_values(by = 'URL', ascending = False)
	filtered_grouped = filtered_grouped.rename(columns={"URL": "count"})
	filtered_grouped_0 = filtered_grouped[filtered_grouped['UND'] == 0]
	filtered_grouped_1 = filtered_grouped[filtered_grouped['UND'] == 1]
	
	channel_titles_0 = np.array(list(filtered_grouped_0['channelTitle']))
	counts_0 = np.array(list(filtered_grouped_0['count']))

	channel_titles_1 = np.array(list(filtered_grouped_1['channelTitle']))
	counts_1 = np.array(list(filtered_grouped_1['count']))

	title_count_0 = {}
	for i, v in enumerate(channel_titles_0):
		title_count_0[v] = counts_0[i]

	title_count_1 = {}
	for i, v in enumerate(channel_titles_1):
		title_count_1[v] = counts_1[i]

	count_l = []
	count_h = []
	top5_channels = channel_titles_1[:5]
	for ch in top5_channels:
		if(ch in channel_titles_1):
			count_h.append(title_count_1[ch])
		else:
			count_h.append(0)
		if(ch in channel_titles_0):
			count_l.append(title_count_0[ch])
		else:
			count_l.append(0)

	# print(count_l)
	# print(count_h)
	# print(top5_channels)

	N = 5
	blue_bar = count_l
	orange_bar = count_h
	ind = np.arange(N)
	plt.figure(figsize=(10,5))
	width = 0.3
	fig = plt.figure()
	fig.subplots_adjust(bottom=0.52)

	plt.bar(ind, blue_bar , width, label='Low Understandability', color = 'darkgrey')
	plt.bar(ind + width, orange_bar, width, label='High Understandability', color = 'goldenrod')

	plt.xlabel('Channel Title')
	plt.ylabel('Count of Videos')
	plt.title('Division of Videos for the Top 5 Channels with the Highest Understandability')
	plt.xticks(ind + width / 2, (top5_channels[0], top5_channels[1], top5_channels[2], top5_channels[3], top5_channels[4]),
				rotation = 90)
	plt.legend(loc='best')
	plt.show()

	# for idx, val in enumerate(channel_titles_1):


	print(filtered_grouped_0)

def views_top5_channels(labeled_df):
	filtered = labeled_df[['channelTitle', 'UND', 'URL']] # channelViewCount
	filtered_grouped = filtered.groupby(['channelTitle', 'UND']).count().reset_index().sort_values(by = 'URL', ascending = False)
	filtered_grouped = filtered_grouped.rename(columns={"URL": "sum"})
	# print(filtered_grouped)
	filtered_grouped_0 = filtered_grouped[filtered_grouped['UND'] == 0]
	filtered_grouped_1 = filtered_grouped[filtered_grouped['UND'] == 1]
	channel_titles_0 = np.array(list(filtered_grouped_0['channelTitle']))
	counts_0 = np.array(list(filtered_grouped_0['sum']))

	channel_titles_1 = np.array(list(filtered_grouped_1['channelTitle']))
	counts_1 = np.array(list(filtered_grouped_1['sum']))

	title_count_0 = {}
	for i, v in enumerate(channel_titles_0):
		title_count_0[v] = counts_0[i]

	title_count_1 = {}
	for i, v in enumerate(channel_titles_1):
		title_count_1[v] = counts_1[i]

	count_l = []
	count_h = []
	top5_channels = channel_titles_1[:5]

	sum0_l = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[0]) & (labeled_df['UND'] == 0)]['viewCount'].sum()
	sum1_l = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[1]) & (labeled_df['UND'] == 0)]['viewCount'].sum()
	sum2_l = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[2]) & (labeled_df['UND'] == 0)]['viewCount'].sum()
	sum3_l = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[3]) & (labeled_df['UND'] == 0)]['viewCount'].sum()
	sum4_l = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[4]) & (labeled_df['UND'] == 0)]['viewCount'].sum()

	sum0_h = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[0]) & (labeled_df['UND'] == 1)]['viewCount'].sum()
	sum1_h = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[1]) & (labeled_df['UND'] == 1)]['viewCount'].sum()
	sum2_h = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[2]) & (labeled_df['UND'] == 1)]['viewCount'].sum()
	sum3_h = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[3]) & (labeled_df['UND'] == 1)]['viewCount'].sum()
	sum4_h = labeled_df.loc[(labeled_df['channelTitle'] == top5_channels[4]) & (labeled_df['UND'] == 1)]['viewCount'].sum()

	count_l = [sum0_l, sum1_l, sum2_l, sum3_l, sum4_l]
	count_h = [sum0_h, sum1_h, sum2_h, sum3_h, sum4_h]

	# print(count_l)
	# print(count_h)
	# print(top5_channels)

	N = 5
	blue_bar = count_l
	orange_bar = count_h
	ind = np.arange(N)
	plt.figure(figsize=(10,5))
	width = 0.3
	fig = plt.figure()
	fig.subplots_adjust(bottom=0.52)

	plt.bar(ind, blue_bar , width, label='Low Understandability', color = 'darkgrey')
	plt.bar(ind + width, orange_bar, width, label='High Understandability', color = 'goldenrod')

	plt.xlabel('Channel Title')
	plt.ylabel('Total Views of all Videos')
	plt.title('Division of Videos for the Top 5 Channels with the Highest Understandability')
	plt.xticks(ind + width / 2, (top5_channels[0], top5_channels[1], top5_channels[2], top5_channels[3], top5_channels[4]),
				rotation = 90)
	plt.legend(loc='best')
	plt.show()


def keyword_wise_und(labeled_df):
	filtered = labeled_df[['UND', 'keyword', 'URL']].groupby(['UND', 'keyword']).count().reset_index().sort_values(by = 'URL', ascending = False)
	filtered_0 = filtered[filtered['UND'] == 0].rename(columns={"URL": "count"})
	filtered_1 = filtered[filtered['UND'] == 1].rename(columns={"URL": "count"})
	keywords_0 = list(filtered_0['keyword'])[:5]
	counts_0 = list(filtered_0['count'])[:5]
	keywords_1 = list(filtered_1['keyword'])[:5]
	counts_1 = list(filtered_1['count'])[:5]

	counts_01 = []
	counts_10 = []

	for k in keywords_0:
		temp = labeled_df.loc[(labeled_df['keyword'] == k) & (labeled_df['UND'] == 1)]['URL'].count()
		counts_01.append(temp)

	for k in keywords_1:
		temp = labeled_df.loc[(labeled_df['keyword'] == k) & (labeled_df['UND'] == 0)]['URL'].count()
		counts_10.append(temp)

	print(keywords_1)
	print(counts_1)
	print(counts_10)

	# fig, axs = plt.subplots(2)
	# fig.suptitle('Most popular keywords in each category')
	# axs[0].bar(keywords_0[:5], counts_0[:5], label = 'Low Understandability', color = 'darkgrey')
	# # plt.legend(loc='best')
	# axs[1].bar(keywords_1[:5], counts_1[:5], label = 'High Understandability', color = 'goldenrod')
	# # plt.legend(loc='best')
	# # handles, labels = axs.get_legend_handles_labels()
	# fig.legend(loc='upper right')
	# plt.show()
	# print(filtered_0)
	N = 5
	blue_bar = counts_1
	orange_bar = counts_10
	ind = np.arange(N)
	plt.figure(figsize=(10,5))
	width = 0.3
	fig = plt.figure()
	fig.subplots_adjust(bottom=0.32)

	plt.bar(ind + width, orange_bar, width, label='Low Understandability', color = 'darkgrey')
	plt.bar(ind, blue_bar , width, label='High Understandability', color = 'goldenrod')
	

	plt.xlabel('Keyword')
	plt.ylabel('Total Count of all Videos')
	plt.title('Count of videos for the most popular keywords')
	plt.xticks(ind + width / 2, (keywords_1[0], keywords_1[1], keywords_1[2], keywords_1[3], keywords_1[4]),
				rotation = 90)
	plt.legend(loc='best')
	plt.show()

def keyword_wise_med(labeled_df):
	filtered = labeled_df[['MED', 'keyword', 'URL']].groupby(['MED', 'keyword']).count().reset_index().sort_values(by = 'URL', ascending = False)
	filtered_0 = filtered[filtered['MED'] == 0].rename(columns={"URL": "count"})
	filtered_1 = filtered[filtered['MED'] == 1].rename(columns={"URL": "count"})
	keywords_0 = list(filtered_0['keyword'])[:5]
	counts_0 = list(filtered_0['count'])[:5]
	keywords_1 = list(filtered_1['keyword'])[:5]
	counts_1 = list(filtered_1['count'])[:5]

	counts_01 = []
	counts_10 = []

	for k in keywords_0:
		temp = labeled_df.loc[(labeled_df['keyword'] == k) & (labeled_df['MED'] == 1)]['URL'].count()
		counts_01.append(temp)

	for k in keywords_1:
		temp = labeled_df.loc[(labeled_df['keyword'] == k) & (labeled_df['MED'] == 0)]['URL'].count()
		counts_10.append(temp)

	print(keywords_1)
	print(counts_1)
	print(counts_10)

	# fig, axs = plt.subplots(2)
	# fig.suptitle('Most popular keywords in each category')
	# axs[0].bar(keywords_0[:5], counts_0[:5], label = 'Low Understandability', color = 'darkgrey')
	# # plt.legend(loc='best')
	# axs[1].bar(keywords_1[:5], counts_1[:5], label = 'High Understandability', color = 'goldenrod')
	# # plt.legend(loc='best')
	# # handles, labels = axs.get_legend_handles_labels()
	# fig.legend(loc='upper right')
	# plt.show()
	# print(filtered_0)
	N = 5
	blue_bar = counts_1
	orange_bar = counts_10
	ind = np.arange(N)
	plt.figure(figsize=(10,5))
	width = 0.3
	fig = plt.figure()
	fig.subplots_adjust(bottom=0.38)

	plt.bar(ind + width, orange_bar, width, label='Low Medical Information', color = 'darkgrey')
	plt.bar(ind, blue_bar , width, label='High Medical Information', color = 'goldenrod')
	

	# plt.xlabel('Keyword')
	plt.ylabel('Total Count of all Videos')
	plt.title('Count of videos for the most popular keywords')
	plt.xticks(ind + width / 2, (keywords_1[0], keywords_1[1], keywords_1[2], keywords_1[3], keywords_1[4]),
				rotation = 90)
	plt.legend(loc='best')
	plt.show()

keyword_wise_med(labeled_df)

med_counts(labeled_df)
views_top5_channels(labeled_df)
channel_wise_count_und(labeled_df)

make_pie_chart(labeled_df)
make_density_plots(labeled_df)

year_wise_counts(labeled_df)


grouped_data = group_data(labeled_df, 'dislikeCount', 'UND')
plot_avg_std_dislike_count('dislikeCount', grouped_data, 'Average Dislike Count for Understandability')

corr_matrix = labeled_df[['UND', 'MED']].corr()
# plt.matshow(corr_matrix, fignum=1)
# plt.show()


plot_best_channels(labeled_df)

def count_UND_for_each_category(labeled_df):
	df = labeled_df[['UND', 'categoryId']].groupby('categoryId').agg(
					  {'UND':['sum', 'count']})
	df = df.reset_index()
	categories = list(df['categoryId'])
	sums = np.array(list(df['UND']['sum']))
	counts = np.array(list(df['UND']['count']))
	props = sums / counts
	new_df = pd.DataFrame(list(zip(categories, counts, props)), columns = ['Category', 'Count', "Proportion"])
	new_df = new_df.sort_values(['Proportion', 'Count'], ascending=[False, False])
	# new_df = new_df.reset_index()
	categories = list(map(str, list(new_df['Category'])))
	props = list(new_df['Proportion'])
	plt.bar(categories, props)
	plt.title('Proportion of videos in each category having High Understandability')
	plt.xlabel('Category ID')
	plt.ylabel('Proportion')
	plt.show()
	# print(new_df)

count_UND_for_each_category(labeled_df)

def counts_for_each_category(labeled_df, col, title):
	df = labeled_df[['UND', col]].groupby('UND').agg(
					  {col:['mean', 'std']})
	
	x = np.array(['Low Understandability', 'High Understandability'])
	y = np.array(list(df[col]['mean']))
	e = np.array(list(df[col]['std']))

	# plt.errorbar(x, y, e, linestyle='None', marker='o', capsize=3)
	fig = plt.figure()
	fig.subplots_adjust(bottom=0.2)
	plt.bar(x, y)
	plt.title(title)
	plt.show()

counts_for_each_category(labeled_df, 'channelSubscriberCount', 'Average Channel Subscriber Count for the Understandability metric')
counts_for_each_category(labeled_df, 'channelViewCount', 'Average Channel View Count for the Understandability metric')
counts_for_each_category(labeled_df, 'viewCount', 'Average number of Views for the Understandability metric')
counts_for_each_category(labeled_df, 'channelCommentCount', 'Average number of comments for the Understandability metric')
counts_for_each_category(labeled_df, 'channelVideoCount', 'Average number of videos in channel for the Understandability metric')
counts_for_each_category(labeled_df, 'likeCount', 'Average number of likes for videos for the Understandability metric')

corr_matrix = labeled_df[['channelSubscriberCount', 'viewCount']].corr()
# print(corr_matrix) # 0.16849



def print_top_words(topic_word_distributions, num_top_words, vectorizer):
	vocab = vectorizer.get_feature_names()
	num_topics = len(topic_word_distributions)
	# print('Displaying the top %d words per topic and their probabilities within the topic...' % num_top_words)
	# print()
	word_prob_list = {}
	for i in range(num_topics):
		word_prob_list[i] = []
	for topic_idx in range(num_topics):
		print('[Topic ', topic_idx, ']', sep='')
		sort_indices = np.argsort(-topic_word_distributions[topic_idx])
		for rank in range(num_top_words):
			word_idx = sort_indices[rank]
			# print(vocab[word_idx], ':',
			# topic_word_distributions[topic_idx, word_idx])
			# print(topic_idx)
			# if(len(word_prob_list[topic_idx])) == 0:
			#     word_prob_list[topic_idx] = []
			# else:
			word_prob_list[topic_idx].append((vocab[word_idx], topic_word_distributions[topic_idx, word_idx]))
		print()
	return(word_prob_list)

year_wise_counts(labeled_df)

def clean_text(corpus_text):
	for c in string.punctuation:
		corpus_text = corpus_text.replace(c, "")  # -- (1)
	
	text = re.sub(r'\S*\d\S*', '', corpus_text) # -- (2)
	text = re.sub(r'[^\w\s]', '', text)         # -- (3)
	
	text = text.lower().split()           # -- (4)         
	
	li = []
	for token in text:
		if(token not in {'video', 'subscribe', 'facebook', 'twitter', 'free'}):
			li.append(token)

	return " ".join(li)

def topic_modeling_low(labeled_df):
	description_df = labeled_df[['description', 'UND']]
	low_understandability = description_df[description_df['UND'] == 0]
	# high_understandability = description_df[description_df['UND'] == 1]
	low_understandability_desc_list = list(low_understandability['description'])
	# high_understandability_desc_list = list(high_understandability['description'])
	low_understandability_desc_list = [clean_text(str(i)) for i in low_understandability_desc_list]
	tf_vectorizer = CountVectorizer(max_df=0.95,
								min_df=0.2,
								stop_words='english',
								max_features=1000)
	tf = tf_vectorizer.fit_transform(low_understandability_desc_list)
	num_topics = 3
	lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
	lda.fit(tf)
	topic_word_distributions = np.array([row / row.sum() for row in lda.components_])
	num_top_words = 8
	print('Displaying the top %d words per topic and their probabilities within the topic for videos that have low understandability' % num_top_words)
	print()
	word_dict = print_top_words(topic_word_distributions, num_top_words, tf_vectorizer)
	return(word_dict)

def topic_modeling_high(labeled_df):
	description_df = labeled_df[['description', 'UND']]
	# low_understandability = description_df[description_df['UND'] == 0]
	high_understandability = description_df[description_df['UND'] == 1]
	# low_understandability_desc_list = list(low_understandability['description'])
	high_understandability_desc_list = list(high_understandability['description'])
	high_understandability_desc_list = [clean_text(str(i)) for i in high_understandability_desc_list]
	tf_vectorizer = CountVectorizer(max_df=0.95,
								min_df=0.2,
								stop_words='english',
								max_features=1000)
	tf = tf_vectorizer.fit_transform(high_understandability_desc_list)
	num_topics = 3
	lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
	lda.fit(tf)
	topic_word_distributions = np.array([row / row.sum() for row in lda.components_])
	num_top_words = 8
	print('Displaying the top %d words per topic and their probabilities within the topic for videos that have high understandability' % num_top_words)
	print()
	word_dict = print_top_words(topic_word_distributions, num_top_words, tf_vectorizer)
	return(word_dict)


word_dict_low = topic_modeling_low(labeled_df)
print('='*50)
word_dict_high = topic_modeling_high(labeled_df)

def show_topic_distribution(topic_id, word_dict_low, word_dict_high):
	relevant_list_low = word_dict_low[topic_id]
	words_low = [word for (word, prob) in relevant_list_low]
	probs_low = [prob for (word, prob) in relevant_list_low]
	low_dict = {}
	for idx, word in enumerate(words_low):
		low_dict[word] = probs_low[idx]

	relevant_list_high = word_dict_high[topic_id]
	words_high = [word for (word, prob) in relevant_list_high]
	probs_high = [prob for (word, prob) in relevant_list_high]
	high_dict = {}
	for idx, word in enumerate(words_high):
		high_dict[word] = probs_high[idx]

	all_words = list(set(words_low + words_high))

	prob_dict = {}
	for word in all_words:
		prob_dict[word] = []
	for idx, word in enumerate(all_words):
		print(idx)
		if(word in words_low):
			prob_dict[word].append(low_dict[word])
		elif(word not in words_low):
			prob_dict[word].append(0)

		if(word in words_high):
			prob_dict[word].append(high_dict[word])
		elif(word not in words_high):
			prob_dict[word].append(0)

	# print(words_low)
	# print(words_high)
	# print(all_words)
	print(prob_dict)
	words = []
	low_probs = []
	high_probs = []
	for word, (low_p, high_p) in prob_dict.items():
		words.append(word)
		low_probs.append(low_p)
		high_probs.append(high_p)
	plt.figure(figsize=(10,5))

	# Width of a bar 
	# width = 0.3

	# ind = np.arange(len(words))
	# Plotting

	X_axis = np.arange(len(words))
	fig = plt.figure()
	plt.bar(X_axis - 0.2, low_probs , 0.4, label='Low understandability')
	plt.bar(X_axis + 0.2, high_probs, 0.4, label='High understandability')
	fig.subplots_adjust(bottom=0.3)

	plt.xticks(X_axis, words, rotation = 90)

	plt.xlabel('Frequent Words')
	plt.ylabel('Probability of word')
	plt.title('Comparing probabilities of words (Low understandability vs High understandability)')

	# xticks()
	# First argument - A list of positions at which ticks should be placed
	# Second argument -  A list of labels to place at the given locations
	# plt.xticks(ind + width / 2, ('Xtick1', 'Xtick3', 'Xtick3'))

	# Finding the best position for legends and putting it
	plt.legend(loc='best')
	# plt.show()
	return(plt)
	# plt.bar(words, probs)
	# plt.show()

plt1 = show_topic_distribution(0, word_dict_low, word_dict_high)
plt2 = show_topic_distribution(1, word_dict_low, word_dict_high)
plt3 = show_topic_distribution(2, word_dict_low, word_dict_high)

plt1.show()
plt2.show()
plt3.show()