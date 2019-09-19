import spacy
from nltk.corpus import state_union

def get_text():
    text = ''
    for file in state_union.fileids():
        text += state_union.raw(file)
    return text


def remove_stop_words(doc, nlp):
    doc_list = [token.text for token in doc if not token.is_stop]

    print(type(doc_list))

    doc_string = " ".join(doc_list)
    doc = nlp(doc_string)
    return doc


def lemmatization(doc, nlp):
    doc_list = [token.lemma_ for token in doc]
    doc_string = " ".join(doc_list)
    doc = nlp(doc_string)
    return doc


def get_bow(doc):
    word_count = {}  # dictionary for the bag of words
    for token in doc:
        if token.is_alpha:
            if token.text not in word_count.keys():
                word_count[token.text] = 1
            else:
                word_count[token.text] += 1

    return word_count


def main():
    text = get_text()
    text = text.lower()

    nlp = spacy.load('en')
    nlp.max_length = len(text)
    print(len(text))

    doc = nlp(text)
    print(len(doc))

    doc = remove_stop_words(doc, nlp)
    doc = lemmatization(doc, nlp)

    word_count = get_bow(doc)

    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    for i in word_count:
        print(i, end='\n')


if __name__ == '__main__':
    main()
