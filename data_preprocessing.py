def get_pairs(lines):
    data = list()
    for line in lines.split('\n'):
        data.append(line.split('\t'))
    return data

#DATA CLEANING

### remove accents
def remove_accents(input):
    s = unicodedata.normalize('NFD', input)
    to_ascii = s.encode('ASCII', 'ignore')
    #decode to utf-8 otherwise type is a byte with b in front of string
    output = to_ascii.decode('UTF-8')
    return output


def start_stop_add(x):
    return '<start> '+x+' <end>'



#####Lower case, trim and remove non-alphanumeric characters######
def normalize(input,IsPredict=False):
    #trim and lower case
    input = remove_accents(input.lower().strip())
    input = re.sub(r"([.!?])", r" \1", input)
    #remove non alpha  chars
    input = re.sub(r"[^a-zA-Z.!?]+", r" ", input)
    #check for multiple whitespaces
    input = re.sub(r"\s+", r" ", input).strip()

    if not IsPredict:
      output = start_stop_add(input)
    else:
      output = input

    return output




def prepare_data():

    #split
    rawdata = get_pairs(text)

    #remove last element i,e just empty list
    rawdata = rawdata[:-1]

    #clean and generate word pairs
    pairs = [[normalize(s) for s in l] for l in rawdata]

    #create cleaned raw data for eng and fre
    raw_data_eng = []
    raw_data_fre = []
    for i in range(len(pairs)):
        raw_data_eng.append(pairs[i][0])
        raw_data_fre.append(pairs[i][1])


    #TOKENIZATION

    tokenizer_eng = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_eng.fit_on_texts(raw_data_eng)
    data_eng = tokenizer_eng.texts_to_sequences(raw_data_eng)
    data_eng = tf.keras.preprocessing.sequence.pad_sequences(data_eng, padding='post')


    tokenizer_fre = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_fre.fit_on_texts(raw_data_fre)
    data_fre = tokenizer_fre.texts_to_sequences(raw_data_fre)
    data_fre = tf.keras.preprocessing.sequence.pad_sequences(data_fre, padding='post')


    #get max lengths of eng and french after padding
    english_len = [len(i) for i in data_eng]
    french_len = [len(i) for i in data_fre]

    max_len_input = max(english_len)
    max_len_target = max(french_len)


    return data_eng,data_fre, raw_data_eng, raw_data_fre, tokenizer_eng, tokenizer_fre, max_len_input, max_len_target
