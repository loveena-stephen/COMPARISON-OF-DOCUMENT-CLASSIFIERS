import nltk, re, glob
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#list of all documents - content and category
document_list_train = []    
document_list_test = [] 

#list of all words in dataset                                                 
all_words = []                                                      

#method to build feature set for each document
def document_features(document):                                        
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

path = 'C:/NLP_Loveena/Doc_Classify - Method2/DataSet/*.sgm'
files=glob.glob(path)
count=0;

#Iterate through each file
for file in files:
    count += 1
    print ("Parsing File " + str(count) + " ...")                                                            
    f = open(file)
    text = f.read()
    f.close()

    #Extract documents. 1000 per file. len(match)=1000
    match = re.findall('<REUTERS.*?</REUTERS>',text, re.DOTALL) 
    
    #Iterate through each document in the this file               
    for item in match:
        
        #Extract document id for each document
        document_id = re.findall('NEWID="(.*?)">',item)    
        
        #Determine if it belongs to the train set or test set
        doc_type = re.findall('LEWISSPLIT="(.*?)" CGISPLIT',item)
        doc_type = str(doc_type[0]) 
                         
        #Extract title for each document
        title = re.findall('<TITLE>(.*?)</TITLE>',item)                        
        if(len(title)>0):
            title = str(title[0])
        else:
            title = ""
            
        #Extract content for each document. Ignore article if content is empty
        body = re.findall('<BODY>(.*?)/BODY>',item,re.DOTALL)                
        if (len(body)>0):
            body = str(body[0]).replace("\n", " ")
        else:
            continue
        
        #convert document content to a list of tokens   
        body_token = word_tokenize(body.lower())            
        #Create list of words after removing stop words and take only words with length>3
        stopwords_list = stopwords.words()
        p = re.compile('[a-zA-Z]+');
        min_length=3
        body_tokens = [SnowballStemmer("porter").stem(word) for word in body_token if word not in stopwords_list]
        body_token_filtered = list(filter(lambda token: p.match(token) and len(token)>=min_length, body_tokens)) 
        
        #Create list of all words in corpus
        all_words += body_token_filtered 
          
        #Extract topic for each document. Ignore article if topic is empty
        categories_tags = re.findall('<TOPICS>(.*?)</TOPICS>',item)
        if(len(categories_tags[0])==0):
            continue
        else:             
            categories = str(categories_tags[0])
            categories_list = re.findall('<D>(.*?)</D>',categories)            
            category = categories_list[0]
            
        #Create a tuple of words in document and category of document
        document_tuple = (body_token_filtered, category) 
          
        #Add the document to it's corresponding list
        if(doc_type=="TRAIN"):              
            document_list_train.append(document_tuple)     
        else:
            document_list_test.append(document_tuple)                                                          
                

#Pick the most common x words in the dataset as features
x=1000
all_words_freq = nltk.FreqDist(all_words)
values = all_words_freq.most_common(x)
word_features = [list(t) for t in zip(*values)][0]  
print("\nNumber of Features Selected: " + str(len(word_features)))                               

#train set and test set are lists where each element is a tuple of two things:
#one - the list of all features indicating if it is present or not in that document
#two - the category of the document
train_set = [(document_features(d), c) for (d,c) in document_list_train]
test_set = [(document_features(d), c) for (d,c) in document_list_test]

print("\nNumber of documents in training set: " + str(len(train_set)))
print("Number of documents in testing set: " + str(len(test_set)))

classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier_accuracy = nltk.classify.accuracy(classifier, test_set)
print("\nClassifier_Accuracy: " + str(classifier_accuracy))
