import csv
import pandas as pd
import math
fields=['statuss','textss']
f=open ("dataset_NB.txt","r")
f1=open("my.txt","w")
li=["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would", "able", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already", "also", "although", "always", "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "arent", "arise", "around", "aside", "ask", "asking", "auth", "available", "away", "awfully", "b", "back", "became", "become", "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "believe", "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "couldnt", "date", "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "former", "formerly", "forth", "found", "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im", "immediate", "immediately", "importance", "important", "inc", "indeed", "index", "information", "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug", "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often", "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall", "owing", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th", "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "theyd", "theyre", "think", "thou", "though", "thoughh", "thousand", "throug", "throughout", "thru", "thus", "til", "tip", "together", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went", "werent", "whatever", "what'll", "whats", "whence", "whenever", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim", "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate", "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning", "consequently", "consider", "considering", "corresponding", "course", "currently", "definitely", "described", "despite", "entirely", "exactly", "example", "going", "greetings", "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second", "secondly", "sensible", "serious", "seriously", "sure", "t's", "third", "thorough", "thoroughly", "three", "well", "wonder", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl", "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "pas", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "par", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs", "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr", "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj", "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on", "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra", "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1", "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"]
for x in f:
	x=x.replace('!',' ')
	x=x.replace('.',' ')
	x=x.replace(',',' ')
	x=x.replace('-',' ')
	x=x.replace(')',' ')
	x=x.replace('(',' ')
	x=x.replace(']',' ')
	x=x.replace('[',' ')
	x=x.replace('{',' ')
	x=x.replace('}',' ')
	x=x.replace('\'',' ')
	x=x.replace('\\',' ')
	x=x.replace('"',' ')
	x=x.replace('\t',' ')
	x=x.replace('/',' ')
	x=x.replace(':',' ')
	x=x.replace(';',' ')
	x=x.replace('?',' ')
	f1.write(x)	
f1.close()	
f1=open("my.txt","r")
list1=[]# nested list
for x in f1:	
	x=x.rstrip()
	y=x[len(x)-1]
	list2=[] # spam or not
	x=x.replace("0",' ')
	x=x.replace("1",' ')
	x=x.replace("2",' ')
	x=x.replace("3",' ')
	x=x.replace("4",' ')
	x=x.replace("5",' ')
	x=x.replace("6",' ')
	x=x.replace("7",' ')
	x=x.replace("8",' ')
	x=x.replace("9",' ')
	z=x.split(" ")
	x=""
	for k in z:
		if k in li:
			k=k.replace(k," ")		
		x+=k
		x+=" "	
	x=x.rstrip()
	if y=='0':
		list2.append('nspam')
	elif y=='1':
	    list2.append('spam')   		
	list2.append(x)
	list1.append(list2)		
with open ("final2.csv","w") as f2:
	writer=csv.writer(f2)
	writer.writerow(fields)
	writer.writerows(list1)	
f3=pd.read_csv("final2.csv")
f3['textss'] = f3['textss'].str.replace('\W+', ' ').str.replace('\s+', ' ').str.strip()
f3['textss'] = f3['textss'].str.lower()
f3['textss'] = f3['textss'].str.split()
alpha=1
def Pspam_word(word): # P(word/spam)
	if word in train_data.columns:
		return (train_data.loc[train_data['statuss']=='spam',word].sum()+alpha)/(No_of_words_in_spam+alpha*No_of_words)
	else:	
		return 1	
def Pnspam_word(word):
    if word in train_data.columns:
        return (train_data.loc[train_data['statuss']=='nspam',word].sum()+alpha)/(No_of_words_in_nonspam+alpha*No_of_words)
    else:
        return 1       
def detect(message):
	spam=1
	nonspam=1
	for k in message:
		spam *= Pspam_word(k)
		spam*=Pspam
		nonspam *= Pnspam_word(k)
		nonspam*=Pnonspam
	if spam > nonspam:
	    return 'spam'
	elif nonspam>spam:
	    return 'nspam'  	
	else:
	    return 'check' 
 #1st fold
 #p(spam_nonsapm/word)=p(word/spam_nonspam)*p(spam_nonspam)/p(getting that word)

test_data=pd.DataFrame(columns=['statuss','textss'])
train_data=pd.DataFrame(columns=['statuss','textss'])
for index,row in f3.iterrows():
	if index<143:
		test_data=test_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
	else:
		train_data=train_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
	     	
total_rows=train_data.shape[0]
no_of_spam=train_data['statuss'].value_counts()['spam']
Pspam=no_of_spam/total_rows
no_of_non_spam=train_data['statuss'].value_counts()['nspam']
Pnonspam=no_of_non_spam/total_rows
total_words=set(train_data['textss'].sum())
word_count =pd.DataFrame([[row[1].count(word) for word in total_words]for index, row in train_data.iterrows()], columns=total_words)
word_count.reset_index(drop=True)
train_data = pd.concat([train_data.reset_index(), word_count], axis=1)
train_data.drop('index',axis='columns', inplace=True)
No_of_words=len(train_data.columns)-2
No_of_words_in_spam=train_data.loc[train_data['statuss']=='spam','textss'].apply(len).sum()
No_of_words_in_nonspam=train_data.loc[train_data['statuss']=='nspam','textss'].apply(len).sum()
train_data.reset_index(drop=True)
test_data['prediction']='spam'
count=0
for index,row in test_data.iterrows():
	k=detect(row[1])
	count = count+1
	test_data.iloc[index,[2]]=k
print("For 1st fold")	
print(test_data)		
accuracy1 = (test_data['prediction'] == test_data['statuss']).sum() / test_data.shape[0] * 100
print(accuracy1)




#2nd fold
test_data=pd.DataFrame(columns=['statuss','textss'])
train_data=pd.DataFrame(columns=['statuss','textss'])

for index,row in f3.iterrows():
	if index>=143 and index<286:
		test_data=test_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
	else:
		train_data=train_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
total_rows=train_data.shape[0]
no_of_spam=train_data['statuss'].value_counts()['spam']
Pspam=no_of_spam/total_rows
no_of_non_spam=train_data['statuss'].value_counts()['nspam']
Pnonspam=no_of_non_spam/total_rows
total_words=set(train_data['textss'].sum())
word_count =pd.DataFrame([[row[1].count(word) for word in total_words]for index, row in train_data.iterrows()], columns=total_words)
word_count.reset_index(drop=True)
train_data = pd.concat([train_data.reset_index(), word_count], axis=1)
train_data.drop('index',axis='columns', inplace=True)
No_of_words=len(train_data.columns)-2
No_of_words_in_spam=train_data.loc[train_data['statuss']=='spam','textss'].apply(len).sum()
No_of_words_in_nonspam=train_data.loc[train_data['statuss']=='nspam','textss'].apply(len).sum()
train_data.reset_index(drop=True)
test_data['prediction']='spam'
count=0
for index,row in test_data.iterrows():
	k=detect(row[1])
	count = count+1
	test_data.iloc[index,[2]]=k
print("For 2nd fold")	
print(test_data)		
accuracy2 = (test_data['prediction'] == test_data['statuss']).sum() / test_data.shape[0] * 100
print(accuracy2)
#3rd fold
test_data=pd.DataFrame(columns=['statuss','textss'])
train_data=pd.DataFrame(columns=['statuss','textss'])
for index,row in f3.iterrows():
	if index>=286 and index<429:
		test_data=test_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
	else:
		train_data=train_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
total_rows=train_data.shape[0]
no_of_spam=train_data['statuss'].value_counts()['spam']
Pspam=no_of_spam/total_rows
no_of_non_spam=train_data['statuss'].value_counts()['nspam']
Pnonspam=no_of_non_spam/total_rows
total_words=set(train_data['textss'].sum())
word_count =pd.DataFrame([[row[1].count(word) for word in total_words]for index, row in train_data.iterrows()], columns=total_words)
word_count.reset_index(drop=True)
train_data = pd.concat([train_data.reset_index(), word_count], axis=1)
train_data.drop('index',axis='columns', inplace=True)
No_of_words=len(train_data.columns)-2
No_of_words_in_spam=train_data.loc[train_data['statuss']=='spam','textss'].apply(len).sum()
No_of_words_in_nonspam=train_data.loc[train_data['statuss']=='nspam','textss'].apply(len).sum()
train_data.reset_index(drop=True)
test_data['prediction']='spam'
count=0
for index,row in test_data.iterrows():
	k=detect(row[1])
	count = count+1
	test_data.iloc[index,[2]]=k
print("For 3rd fold")	
print(test_data)		
accuracy3 = (test_data['prediction'] == test_data['statuss']).sum() / test_data.shape[0] * 100
print(accuracy3)
#4th fold
test_data=pd.DataFrame(columns=['statuss','textss'])
train_data=pd.DataFrame(columns=['statuss','textss'])
for index,row in f3.iterrows():
	if index>=429 and index<572:
		test_data=test_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
	else:
		train_data=train_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
total_rows=train_data.shape[0]
no_of_spam=train_data['statuss'].value_counts()['spam']
Pspam=no_of_spam/total_rows
no_of_non_spam=train_data['statuss'].value_counts()['nspam']
Pnonspam=no_of_non_spam/total_rows
total_words=set(train_data['textss'].sum())
word_count =pd.DataFrame([[row[1].count(word) for word in total_words]for index, row in train_data.iterrows()], columns=total_words)
word_count.reset_index(drop=True)
train_data = pd.concat([train_data.reset_index(), word_count], axis=1)
train_data.drop('index',axis='columns', inplace=True)
No_of_words=len(train_data.columns)-2
No_of_words_in_spam=train_data.loc[train_data['statuss']=='spam','textss'].apply(len).sum()
No_of_words_in_nonspam=train_data.loc[train_data['statuss']=='nspam','textss'].apply(len).sum()
train_data.reset_index(drop=True)
test_data['prediction']='spam'
count=0
for index,row in test_data.iterrows():
	k=detect(row[1])
	count = count+1
	test_data.iloc[index,[2]]=k
print("For 4th fold")	
print(test_data)		
accuracy4 = (test_data['prediction'] == test_data['statuss']).sum() / test_data.shape[0] * 100
print(accuracy4)
#5th fold
test_data=pd.DataFrame(columns=['statuss','textss'])
train_data=pd.DataFrame(columns=['statuss','textss'])
for index,row in f3.iterrows():
	if index>=572 and index<715:
		test_data=test_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
	else:
		train_data=train_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
total_rows=train_data.shape[0]
no_of_spam=train_data['statuss'].value_counts()['spam']
Pspam=no_of_spam/total_rows
no_of_non_spam=train_data['statuss'].value_counts()['nspam']
Pnonspam=no_of_non_spam/total_rows
total_words=set(train_data['textss'].sum())
word_count =pd.DataFrame([[row[1].count(word) for word in total_words]for index, row in train_data.iterrows()], columns=total_words)
word_count.reset_index(drop=True)
train_data = pd.concat([train_data.reset_index(), word_count], axis=1)
train_data.drop('index',axis='columns', inplace=True)
No_of_words=len(train_data.columns)-2
No_of_words_in_spam=train_data.loc[train_data['statuss']=='spam','textss'].apply(len).sum()
No_of_words_in_nonspam=train_data.loc[train_data['statuss']=='nspam','textss'].apply(len).sum()
train_data.reset_index(drop=True)
test_data['prediction']='spam'
count=0
for index,row in test_data.iterrows():
	k=detect(row[1])
	count = count+1
	test_data.iloc[index,[2]]=k
print("For 5th fold")	
print(test_data)		
accuracy5 = (test_data['prediction'] == test_data['statuss']).sum() / test_data.shape[0] * 100
print(accuracy5)
#6th fold
test_data=pd.DataFrame(columns=['statuss','textss'])
train_data=pd.DataFrame(columns=['statuss','textss'])
for index,row in f3.iterrows():
	if index>=715 and index<858:
		test_data=test_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
	else:
		train_data=train_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
total_rows=train_data.shape[0]
no_of_spam=train_data['statuss'].value_counts()['spam']
Pspam=no_of_spam/total_rows
no_of_non_spam=train_data['statuss'].value_counts()['nspam']
Pnonspam=no_of_non_spam/total_rows
total_words=set(train_data['textss'].sum())
word_count =pd.DataFrame([[row[1].count(word) for word in total_words]for index, row in train_data.iterrows()], columns=total_words)
word_count.reset_index(drop=True)
train_data = pd.concat([train_data.reset_index(), word_count], axis=1)
train_data.drop('index',axis='columns', inplace=True)
No_of_words=len(train_data.columns)-2
No_of_words_in_spam=train_data.loc[train_data['statuss']=='spam','textss'].apply(len).sum()
No_of_words_in_nonspam=train_data.loc[train_data['statuss']=='nspam','textss'].apply(len).sum()
train_data.reset_index(drop=True)
test_data['prediction']='spam'
count=0
for index,row in test_data.iterrows():
	k=detect(row[1])
	count = count+1
	test_data.iloc[index,[2]]=k
print("For 6th fold")	
print(test_data)		
accuracy6 = (test_data['prediction'] == test_data['statuss']).sum() / test_data.shape[0] * 100
print(accuracy6)
#7th fold
test_data=pd.DataFrame(columns=['statuss','textss'])
train_data=pd.DataFrame(columns=['statuss','textss'])
for index,row in f3.iterrows():
	if index>=853:
		test_data=test_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
	else:
		train_data=train_data.append({'statuss':row[0],'textss':row[1]},ignore_index=True)
total_rows=train_data.shape[0]
no_of_spam=train_data['statuss'].value_counts()['spam']
Pspam=no_of_spam/total_rows
no_of_non_spam=train_data['statuss'].value_counts()['nspam']
Pnonspam=no_of_non_spam/total_rows
total_words=set(train_data['textss'].sum())
word_count =pd.DataFrame([[row[1].count(word) for word in total_words]for index, row in train_data.iterrows()], columns=total_words)
word_count.reset_index(drop=True)
train_data = pd.concat([train_data.reset_index(), word_count], axis=1)
train_data.drop('index',axis='columns', inplace=True)
No_of_words=len(train_data.columns)-2
No_of_words_in_spam=train_data.loc[train_data['statuss']=='spam','textss'].apply(len).sum()
No_of_words_in_nonspam=train_data.loc[train_data['statuss']=='nspam','textss'].apply(len).sum()
train_data.reset_index(drop=True)
test_data['prediction']='spam'
count=0
for index,row in test_data.iterrows():
	k=detect(row[1])
	count = count+1
	test_data.iloc[index,[2]]=k
print("For 7th fold")	
print(test_data)		
accuracy7 = (test_data['prediction'] == test_data['statuss']).sum() / test_data.shape[0] * 100
print(accuracy7)
average=(accuracy7+accuracy6+accuracy5+accuracy4+accuracy3+accuracy2+accuracy1)/7
print("Overall average accuracy= ")
print(average)