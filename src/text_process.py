from bs4 import BeautifulSoup
import string

def remove_html(text_data):
    soup = BeautifulSoup(text_data, 'lxml')
    return soup.get_text()


def remove_punctuation(text):
    sent = []
    for t in text.split(' '):
        no_punct = "".join([c for c in t if c not in string.punctuation])
        sent.append(no_punct)
        
    sentence = " ".join(s for s in sent)
    return sentence
