print('Load Modul')
from nltk.collocations import *
import nltk
from collections import Counter
import pandas as pd

# ----------------------- DEFINISI FUNGSI ------------------------------
def filter_words(sentence):
	char = "'!$%^&*()_-+=`~,"
	for c in char:
		sentence = sentence.replace(c,' ')
	sentence = sentence.replace('"',' ')
	sentence = sentence.replace("\n"," ")
	sentence = sentence.replace("  "," ")
	return sentence

def stopword(sentence, file_csv, column):
	dfsw = pd.read_csv(file_csv)
	for index, row in dfsw.iterrows():
		sentence = sentence.replace(row[column]," ")
	return sentence
print('Load Modul Completed')
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

hitung = 0

#Keyword and Alias KEBALIK YA
dictionary = {}
dictionary_bigram = {}

#List dari alias
list_alias = []
list_alias_unigram = []
list_alias_bigram = []
print('Load Convo Completed')



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
	tokens = tokens_filter.split()

	bigrams = list(nltk.bigrams(conversation.split()))
	bigrams =  " ".join(map("".join,bigrams))
	bigrams = filter_words(bigrams).split()

	# ----------------------------------------------
	#	DICTIONARY UNIGRAM DAN BIGRAM TERBALIK!!!!
	#dictionary_bigram[alias] = keyword
	#dictionary[keyword] = alias
	# ----------------------------------------------

	#Hasgtag Mention Ectractor
	caption_clean = conversation.replace('#',' #').replace('@', ' @').replace('%','')
	words = caption_clean.split()
	for word in words:
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

	for keyword in dictionary:
		if keyword in tokens:
			list_alias_per_convo.append(dictionary[keyword])
			list_alias.append(dictionary[keyword])
			list_alias_unigram.append(dictionary[keyword])
	for keyword in dictionary_bigram:
		if keyword in bigrams:
			list_alias_per_convo.append(keyword)
			list_alias.append(keyword)
			list_alias_bigram.append(keyword)

		for i in dictionary_bigram:
			if i in tokens:
				list_alias_per_convo.append(keyword)
				list_alias.append(keyword)
				list_alias_bigram.append(keyword)

	for i in range(len(tokens) - 1):
		df_dirty_word = df_dirty_word.append(pd.DataFrame([['unigram',tokens[i]]],columns=['type','word']))

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
					df_word_extraction = df_word_extraction.append(pd.DataFrame([[conversation_raw, w1, w2, w1 + w2]],columns=['conversation','w1','w2','w1w2']))


# ----------------------- PRINT INTO CSV FILE ------------------------------
# Membuat CSV DIRTY WORD
df_dirty_word_count = df_dirty_word.groupby(['word']).size().reset_index(name='counts')
df_dirty_word_count.drop_duplicates(['word','counts']).to_csv('dirty_word_count.csv')

# MEMBUAT CSV CLEAN WORD
counter = Counter(list_alias)
df_clean_word = pd.DataFrame.from_dict(counter, orient='index').reset_index()
df_clean_word.columns = ['convo','counts']
df_clean_word.to_csv('clean_word_count.csv')

# MEMBUAT CSV WORD EXTRACTION
df_word_extraction.to_csv('word_extraction.csv')
df_word_extraction_count = df_word_extraction.groupby(['w1w2','w1','w2']).size().reset_index(name='counts')


# Loop perhitungan Edge BRN
for index, row in df_word_extraction_count.iterrows():
	w1 = row['w1']
	w2 = row['w2']
	w1w2 = row['w1w2']
	w1w2_counts = int(row['counts'])
	w1_counts = int(counter[w1])
	w2_counts = int(counter[w2])
	edge = str((w1_counts * w2_counts) / w1w2_counts)
	df_brn_input = df_brn_input.append(pd.DataFrame([[ w1, w1_counts, w2, w2_counts, w1w2, w1w2_counts, edge]],columns=['w1', 'w1_counts','w2', 'w2_counts','w1w2', 'w1w2_counts','edge'])) 

# MEMBUAT CSV BRN
df_brn_input.to_csv('BRN_Input.csv')
	




