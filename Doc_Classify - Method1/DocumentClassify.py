import re, glob, os
from collections import Counter
from collections import OrderedDict
from whoosh.fields import Schema, STORED, TEXT, NUMERIC
from whoosh.qparser import QueryParser
from whoosh import index
from whoosh import scoring
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


stopwords_list = stopwords.words()
p = re.compile('[a-zA-Z]+')
min_length=3
fileCount=0   
item_count = 0
match_count = 0 

#method to parse and extract information from an article
#Input: An article in XML format bound by the tags <REUTERS> and </REUTERS>
#Output: A list 
#Elements of List
#0 - Article ID
#1 - Title
#2 - Body
#3 - Tokenized and Processed Body
#4 - Topic List
def parse_item(single_article):
    #Extract Article Id
    article_id = re.findall('NEWID="(.*?)">',item)
    
    #Extract Title
    title = re.findall('<TITLE>(.*?)</TITLE>',single_article)
    if(len(title)>0):
        title = str(title[0])
    else:
        title = ""
    
    #Extract body
    body = re.findall('<BODY>(.*?)/BODY>',single_article,re.DOTALL)
    if (len(body)>0):
        body = str(body[0]).replace("\n", " ")
    else:
        body = ""
    
    #convert document content to a list of tokens   
    body_token = word_tokenize(body.lower())            
    #Create list of words after removing stop words and take only words with length>3
    body_tokens = [SnowballStemmer("porter").stem(word) for word in body_token if word not in stopwords_list]
    body_token_filtered = list(filter(lambda token: p.match(token) and len(token)>=min_length, body_tokens))
    
    #Extract Categories
    category = re.findall('<TOPICS>(.*?)</TOPICS>',item)
    category = str(category[0])
    categories_list = re.findall('<D>(.*?)</D>',category)
    
    return_data = [article_id, title, body, body_token_filtered,categories_list]    
    return return_data

#Build Index    
schema = Schema(articleId = NUMERIC(stored=True),
                title = STORED,
                category = STORED,
                content = TEXT,
                processed_content = TEXT)

if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
    
ix = index.create_in("indexdir",schema)

writer = ix.writer()

path = 'C:/NLP_Loveena/Doc_Classify - Method1/DataSetIndexing/*.sgm'
files=glob.glob(path)

for file in files:
    fileCount += 1
    print ("Parsing File " + str(fileCount) + " ...")
   
    f = open(file)
    text = f.read()
    f.close()
    #Extract all article in file
    match = re.findall('<REUTERS.*?</REUTERS>',text, re.DOTALL)
    
    for item in match:
        #Call method to parse each article and extract required information
        parsed_data = parse_item(item) 
        
        article_id = parsed_data[0]
        title = parsed_data[1]
        body = parsed_data[2]
        processed_content = ' '.join(parsed_data[3])
        categories = parsed_data[4]
        
        #Write to Index
        writer.add_document(articleId = article_id,
                            title = title,
                            category = categories,
                            content = body,
                            processed_content = processed_content)    

       
print ("\nCreating Index ...")        
writer.commit()
print ("Index Created\n")

#Read data to classify
path = 'C:/NLP_Loveena/Doc_Classify - Method1/DataSetIndexing/TestSet.txt'
with open(path) as f:
    text = f.read()

#Extract all articles
match = re.findall('<REUTERS.*?</REUTERS>',text, re.DOTALL)

for item in match:
    #Call method to parse each article and extract required information
    parsed_data = parse_item(item)
    if(parsed_data[3]==[] or parsed_data[4]==[]):
        continue
    
    item_count += 1
    
    article_id = parsed_data[0]
    print("Article ID: " + article_id[0])
    
    freq_words = Counter(parsed_data[3]).most_common(10)
    #Pick the most frequent word as the search word
    search_word = freq_words[0][0]
      
    original_category = parsed_data[4][0]
    print("Original Category: " + original_category)
    
    #Query the index for the search word to retrieve all articles that contain that word
    qp = QueryParser("processed_content", schema=ix.schema)
    q = qp.parse(search_word)
    
    #Retrieve the topics of all the articles returned by the search    
    category_list = []
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
        results = searcher.search(q, limit=None)        
        for index in range(0,len(results)):
            category_list.append(results[index]['category'])
    
    category_list = sum(category_list,[])     
    #Pick the most frequently occurring category   
    ordered_category_freq = OrderedDict(sorted(dict(Counter(category_list)).items(), key=lambda v:v[1], reverse=True))
    for key in ordered_category_freq:
        new_category = key
        if(not(new_category is '')):
            break
    
    print("Classified Category: " + new_category + "\n")
    
    #Keep track of articles classified correctly
    if (original_category == new_category):
        match_count += 1

print("Number of articles classified: " + str(item_count))
print("Number of articled classified correctly: " + str(match_count))
match_percentage = (match_count/item_count)*100
print("Match Accuracy: " + str(match_percentage))
