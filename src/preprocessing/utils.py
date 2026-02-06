def keep_only_part(txt_list, number):
    #A utiliser AVANT remove_ponctuation() sinon on peut pas reconnaitre les lignes

    if number is None:
        return txt_list
        
    tmp = txt_list
    for i in range(len(tmp)):
        tmp[i] = tmp[i].split('\n')
        if number > len(tmp[i]):
            print(f"error number{number} greater then length of text{len(tmp[i])}")
            return
        if number > 0:
            tmp[i] = '\n'.join(tmp[i][:number])
        else:
            tmp[i] = '\n'.join(tmp[i][number:])

    return tmp
    

def remove_caps(txt_list):

    tmp = txt_list
    for i in range(len(txt_list)):
        tmp[i].lower()

    return tmp

def remove_ponctuation(txt_list):

    tmp = txt_list
    punc = string.punctuation
    #print(punc)
    punc += '\n\r\t'
    for i in range(len(txt_list)):
        #tmp[i] = re.sub(r"\b's\b", '', tmp[i])
        #tmp[i] = tmp[i].translate(str.maketrans(punc, ' ' * len(punc)))
        tmp[i] = tmp[i].translate(str.maketrans('', '', punc))

    return tmp

def stemming(txt_list):

    ps = PorterStemmer()
    tmp = txt_list
    for i in range(len(txt_list)):
        tmp[i] = ' '.join([ps.stem(word) for word in tmp[i].split()])
    #test = ps.stem("isnt")
    #print(test)
    return tmp
    
def change_capital_words(txt_list):

    tmp = txt_list
    for i in range(len(txt_list)):
        #if re.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', tmp[i]):
            #print("title found", re.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', tmp[i]))
        tmp[i] = re.sub(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', 'TITLE',tmp[i])

    return tmp

def remove_numbers(txt_list):

    tmp = txt_list
    for i in range(len(tmp)):

        tmp[i] = re.sub('[0-9]+', '', tmp[i])

    return tmp

def vectorizer(txt_list, language):

    assert (language == 'FRENCH' or language == 'ENGLISH'), "Language value needs to be either FRENCH or ENGLISH"
    
    if language == "FRENCH":
        stop_list = stopwords.words('french')
    elif language == "ENGLISH":
        stop_list = stopwords.words('english')

    vectorizer = CountVectorizer(stop_words=stop_list)
    X = vectorizer.fit_transform(txt_list)

    return X, vectorizer