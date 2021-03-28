#####read data########

def read_file(filename):
    #Read file that preserves formatting of characters, utf-8 is commonly used to read multiple languages
    lines = open(filename, encoding='utf-8')
    text = lines.read()
    return text
