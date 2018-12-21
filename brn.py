time_start = time.time()
print('Memulai')
from nltk.collocations import *
import nltk
from collections import Counter
import pandas as pd
import time



# ----------------------- DEFINISI FUNGSI ------------------------------
def filter_words(sentence):
	char = "!$%^&*()_-+=.`~,ã:"

	for c in char:
		sentence = sentence.replace(c,' ')
	sentence = sentence.replace('"',' ')
	sentence = sentence.replace("\n"," ")
	sentence = sentence.replace("/"," ")
	sentence = sentence.replace("'"," ")
	sentence = sentence.replace("\\"," ")
	sentence = sentence.replace("  "," ")

	return sentence

def stopword(sentence, file_csv, column):
	dfsw = pd.read_csv(file_csv)
	for index, row in dfsw.iterrows():
		sentence = sentence.replace(row[column]," ")
	return sentence
print('Load Convo')

# ----------------------- LOAD DATA ------------------------------
dfconvo = pd.read_csv('convo.csv')
dfdict = pd.read_csv('dict.csv')

df_dirty_word = pd.DataFrame()
df_dirty_word_bigram = pd.DataFrame()
df_clean_word = pd.DataFrame()
df_word_extraction = pd.DataFrame()
df_brn_input = pd.DataFrame()

bigram = {}

hit = 0
hitung = 0

#Keyword and Alias KEBALIK YA
dictionary = {}
dictionary_bigram = {}

#List dari alias
list_alias = []
list_alias_unigram = []
list_alias_bigram = []


#list_dirty_word
df_dirty_word_list = []


print('Load Convo Completed')

dict_word_extraction = []

# ----------------------- LOAD KEYWORD AND ALIAS  ------------------------------
# Looping untuk memisah unigram dan bigram dari keyword
for index, row in dfdict.iterrows():
	keyword = row['keywords'].lower()
	alias =  row['alias'].lower()
	if ' ' in keyword:
		if alias not in dictionary_bigram:
				dictionary_bigram[alias] = keyword
	else:
		if keyword not in dictionary:
				dictionary[keyword] = alias


# ----------------------- MENGOLAH DATA ------------------------------
# Looping pengolahan convo
for index, row in dfconvo.iterrows():
	list_alias_per_convo = []
	conversation_raw = str(row['conversation']).lower()
	conversation = str(row['conversation']).lower()
	postid = row['postid']
	tokens_filter = filter_words(conversation)
	#Kode dibawah untuk pake stopword
	#tokens_stopword = stopword(conversation, 'stopwords.csv', 'words')

	# tokens = convo yang sudah di tokenize menjadi UNIGRAM (list)
	# bigrams = convo yang sudah di pecah menjadi BIGRAM (list)
	tokens = tokens_filter.split()

	bigrams = list(nltk.bigrams(conversation.split()))
	bigrams =  " ".join(map("".join,bigrams))
	bigrams = filter_words(bigrams).split()



	#Hasgtag Mention Ectractor
	caption_clean = filter_words(conversation).replace('#',' #').replace('@', ' @').replace('%','')
	words = caption_clean.split()
	for word in words:
		if len(word.replace('#','').replace('@','')) != 0:
			if '#' in word:
				a = word.replace("#","")
				list_alias.append(a)
				list_alias_per_convo.append(a)
				conversation = conversation.replace(word,' ')
			if '@' in word:
				a = word.replace("@","")
				list_alias.append(a)
				list_alias_per_convo.append(a)
				conversation = conversation.replace(word,' ')
	# ----------------------------------------------
	#	DICTIONARY UNIGRAM DAN BIGRAM TERBALIK!!!!
	#dictionary_bigram[alias] = keyword
	#dictionary[keyword] = alias
	# ----------------------------------------------
	for keyword in dictionary:
		if keyword in tokens:
			list_alias_per_convo.append(dictionary[keyword])
			list_alias.append(dictionary[keyword])
			list_alias_unigram.append(dictionary[keyword])

	for alias in dictionary_bigram:
		if alias in bigrams:
			list_alias_per_convo.append(alias)
			list_alias.append(alias)
			list_alias_bigram.append(alias)
		elif alias in tokens:
			list_alias_per_convo.append(alias)
			list_alias.append(alias)
			list_alias_bigram.append(alias)

	for i in range(len(tokens)):
		df_dirty_word_list.append(tokens[i])


	bigram_all = {}
	hitung = 0

	#Membuat bigram
	anti_duplicate = []
	for	w1 in list_alias_per_convo:
		for w2 in list_alias_per_convo:
			if w1 != w2:
				if w1 + w2 not in anti_duplicate:
					anti_duplicate.append(w1 + w2)
					anti_duplicate.append(w2 + w1)
					#labelf = 'conversation,w1,w2,w1w2\n'
					#linef = conversation_raw.replace(",",".").replace("\n"," ").replace("\r"," ") + "," + w1 + "," + w2 + "," + w1 + w2 + "\n"
					#f.write(linef)
					dict_word_extraction.append({'1_conversation': conversation_raw, '2_w1': w1, '3_w2': w2, '4_w1w2': w1 + w2})
					#df_word_extraction = df_word_extraction.append(pd.DataFrame([[conversation_raw, w1, w2, w1 + w2]],columns=['conversation','w1','w2','w1w2']))
	print(str(hit) + '. ' + str(time.time() - time_start) + ' detik.' )
	hit += 1

#f.close()
# ----------------------- PRINT INTO CSV FILE ------------------------------
# Membuat CSV DIRTY WORD
df_dirty_word['word'] = pd.Series(df_dirty_word_list).values
df_dirty_word_count = df_dirty_word.groupby(['word']).size().reset_index(name='counts')
df_dirty_word_count.drop_duplicates(['word','counts']).to_csv('dirty_word_count.csv')

print('Membuat CSV Dirty Word Selesai. ' + str(time.time() - time_start) + ' detik.' )

# MEMBUAT CSV CLEAN WORD
counter = Counter(list_alias)
df_clean_word = pd.DataFrame.from_dict(counter, orient='index').reset_index()
df_clean_word.columns = ['convo','counts']
df_clean_word.to_csv('clean_word_count.csv')

print('Membuat CSV Clean Word Selesai. ' + str(time.time() - time_start) + ' detik.' )
# MEMBUAT CSV WORD EXTRACTION
	#df_word_extraction.to_csv('word_extraction.csv')
df_word_extraction = pd.DataFrame.from_dict(dict_word_extraction)
#df_word_extraction = pd.read_csv('word_extraction.csv', sep=',')
df_word_extraction_count = df_word_extraction.groupby(['4_w1w2','2_w1','3_w2']).size().reset_index(name='counts')

print('Membuat CSV Word Extractor Selesai. ' + str(time.time() - time_start) + ' detik.' )
# Loop perhitungan Edge BRN
dict_brn_input = []
hit_loop_brn = 0
for index, row in df_word_extraction_count.iterrows():
	w1 = row['2_w1']
	w2 = row['3_w2']
	w1w2 = row['4_w1w2']
	w1w2_counts = int(row['counts'])
	w1_counts = int(counter[w1])
	w2_counts = int(counter[w2])
	edge = str((w1_counts * w2_counts) / w1w2_counts)
	dict_brn_input.append({'1_w1': w1, '2_w1_counts': w1_counts, '3_w2': w2, '4_w2_counts':w2_counts, '5_w1w2': w1w2, '6_w1w2_counts': w1w2_counts, '7_edge': edge })
	
	print('loop ke ' + str(hit_loop_brn) + ' ' + str(time.time() - time_start) + ' detik.')
	hit_loop_brn += 1
	#df_brn_input = df_brn_input.append(pd.DataFrame([[ w1, w1_counts, w2, w2_counts, w1w2, w1w2_counts, edge]],columns=['w1', 'w1_counts','w2', 'w2_counts','w1w2', 'w1w2_counts','edge'])) 
df_brn_input = pd.DataFrame.from_dict(dict_brn_input)
df_brn_input.columns = ['w1', 'w1_counts', 'w2', 'w2_counts', 'w1w2', 'w1w2_counts', 'edge']

print('Loop Perhitungan Selesai. ' + str(time.time() - time_start) + ' detik.' )
# MEMBUAT CSV BRN
df_brn_input.to_csv('BRN_Input.csv')
print('Membuat CSV BRN Selesai. ' + str(time.time() - time_start) + ' detik.' )




