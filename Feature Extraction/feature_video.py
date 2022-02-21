import glob
import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import readability 
import os
import nltk
from collections import Counter
from statistics import mean 

filepath='/Users/xiaoliu/Desktop/YouTube/CloudVision/' 
summary = ["finally", "in a word", "in brief", "briefly", "in conclusion", "in the end", "in the final analysis", "on the whole", "to conclude", "to summarize", "in sum", "to sum up", "in summary", "lastly"]
transition = ["accordingly", "as a result", "and so", "because", "consequently", "for that reason", "hence", "on account of", "since", "therefore", "thus", "after", "afterwards", "always", "at length", "during", "earlier", "following", "immediately", "in the meantime", "later", "never", "next", "once", "simultaneously", "so far", "sometimes", "soon", "subsequently", "then", "this time", "until now", "when", "whenever", "while", "additionally", "again", "also", "and", "or", "not", "besides", "even more", "finally", "first", "firstly", "further", "furthermore", "in addition", "in the first place", "in the second place", "last", "lastly", "moreover", "next", "second", "secondly", "after all", "although", "and yet", "at the same time", "but", "despite", "however", "in contrast", "nevertheless", "notwithstanding", "on the contrary", "on the other hand", "otherwise", "thought", "yet", "as an illustration", "e.g.", "for example", "for instance", "specifically", "to demonstrate", "to illustrate", "briefly", "critically", "foundationally", "more importantly", "of less importance", "primarily", "above", "centrally", "opposite to", "adjacent to", "below", "peripherally", "nearby", "beyond", "in similar fashion", "in the same way", "likewise", "in like manner", "i.e.", "in other word", "that is", "to clarify", "to explain", "in fact", "of course", "undoubtedly", "without doubt", "surely", "indeed", "for this purpose", "so that", "to this end", "in order that", "to that end"]

for filename in os.listdir('/Users/xiaoliu/Desktop/YouTube/CloudVision/'):
	with open(os.path.join('/Users/xiaoliu/Desktop/YouTube/CloudVision/', filename), 'r') as f:
		lines=f.readlines()
		for line in lines: 
			video=json.loads(line)
			#print(video)
			features={}
			features['id']=video['video_id'][:-4]
			narratives = ' '. join(video["speech_transcription"]["transcription"]["transcript"])
			transition_words = 0
			summary_words = 0
			active_verb=0
			features['readability']=0
			features['sentence_count']=0
			features['word_count']=0
			features['word_unique']=0
			#readability
			text= sent_tokenize(narratives)
			if len(text)>0:
				try:
					results = readability.getmeasures(text, lang='en')
					features['readability']= results['readability grades']['ARI']
					#Sentence count 
					features['sentence_count']=results['sentence info']['sentences']
					#Word count
					features['word_count']=results['sentence info']['words']
					#unique word count
					features['word_unique']=results['sentence info']['wordtypes']
					#active verb 
					tokens=word_tokenize(narratives.lower())
					tags = nltk.pos_tag(tokens)
					counts = Counter( tag for word,  tag in tags)
					active_verb = counts.get("VB",0)+counts.get("VBD",0)+counts.get("VBG",0)+counts.get("VBP",0)+counts.get("VBZ",0)
					#summary words
					for i in summary:
						if i in narratives:
							summary_words+=1
					# transition words
					for i in transition:
						if i in narratives:
							transition_words+=1
				except ValueError:
					print(text)
			features['transition_words']=transition_words
			features['summary_words']=summary_words
			features['active_verb']=active_verb
			if video.get('text_detection')!= None:
				#print(video['text_detection'])
				if len(sum(video['text_detection']['text_segment']['confidence'],[]))>0:
					features['OCR_confidence']= mean(sum(video['text_detection']['text_segment']['confidence'],[]))
			else:
				features['OCR_confidence']=0
			if video.get("speech_transcription")!=None:
				if len(video["speech_transcription"]["transcription"]["confidence"])>0:
					features['transcript_confidence']= mean(video["speech_transcription"]["transcription"]["confidence"])
			else: 
				features['transcript_confidence']=0
			features['shot_count']= len(video['label_detection']['segment_Label_Annotations']['entity_id'])
			if len(sum(video['label_detection']['segment_Label_Annotations']['segment']['confidence'],[]))>0:
				features['shotchange_confidence']= mean(sum(video['label_detection']['segment_Label_Annotations']['segment']['confidence'],[]))
			else:
				features['shotchange_confidence']=0
			print(features)


