import xml.etree.ElementTree as etree
import pickle
import glob
import re
from argparse import ArgumentParser
from nltk.corpus import wordnet as wn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("dataset_name", nargs='?', default='../resources/semcor.data.xml', help="Name of the xml file to use")
    parser.add_argument("gold_name", nargs='?', default='gold2dic', help="Name of the gold2dic file to use")

    return parser.parse_args()

# Save dictionary to file
def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def fngold2dic(filepath="../resources/semcor+omsti.gold.key.txt"):
    mapping = {}

    with open(filepath) as f:
        lines = f.readlines()

        for l in lines:
            map = l.split()
            mapping[map[0]] = map[1]

    save(mapping, '../resources/gold2dic')
    return mapping

def create_dictionary(dataset_name, gold2dic):
    words = []
    wordnet = []

    # get an iterable
    context = etree.iterparse(dataset_name)

    sentence = []
    sentenceNet = []
    deletedSentences = 0

    for event, elem in iter(context):

        if elem.tag == "sentence":# and event == 'start':
            print(elem.attrib['id'])
            if len(words)<5:
                deletedSentences += 1
                words.append(' '.join(sentence))
                wordnet.append(' '.join(sentenceNet))
            sentence = []
            sentenceNet = []

        elif elem.tag == "wf" or elem.tag == "instance":# and event == 'start':
            #word = {}
            #word['lemma'] = elem.attrib["lemma"].lower()
            #word['text'] = elem.text.lower()
            word = elem.attrib["lemma"].lower()
            sentence.append(word)

            if elem.tag == "instance":
                dataset_id = elem.attrib["id"]
                #word['id'] = dataset_id
                synset = wn.lemma_from_key(gold2dic[dataset_id]).synset()
                synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                #word['wordnet'] = synset_id
                sentenceNet.append(synset_id)
            else:
                sentenceNet.append(word)
            #sentence.append(word)

        elem.clear()

    save(words, '../resources/' + 'words')
    save(wordnet, '../resources/' + 'wordnet')

    print('\nSentences removed:', deletedSentences)

    return words, wordnet

def load_data(dataset_name, path="../resources/", gold_name='gold2dic', sentence_size = 100):

    # Check if gold2dic exists
    if glob.glob(path + gold_name + '.pkl'):
        print('\nMapping found!')
        gold2dic = load(path + gold_name)
    else:
        print('\nMapping not found!')
        print('\nBuilding mapping from file')
        gold2dic = fngold2dic()

    # Check if sentences exists
    if glob.glob(path + 'words' + '.pkl') and glob.glob(path + 'wordnet' + '.pkl'):
        print('\nSentences found!')
        words = load(path + 'words')
        wordnet = load(path + 'wordnet')
    else:
        words, wordnet = create_dictionary(dataset_name, gold2dic)

    numberSentences = len(words)
    concatenated = words + wordnet

    # Character dictionary
    word2id = dict()
    word2id["<PAD>"] = 0 #zero is not casual!
    word2id["<UNK>"] = 1 #OOV are mapped as <UNK>
    id = 2

    maxsentence = 0
    minsentence = 10000
    wordtokens = 0
    sensetokens = 0
    wordsIds = []
    wordnetIds = []

    # Tokenizer
    t = Tokenizer(filters='!"#$%&()*+,-./;<=>?@[\\]^_`{|}~\'\t\n')

    t.fit_on_texts(words)
    t.fit_on_texts(wordnet)

    # print(t.word_counts)
    # print(t.document_count)
    # print(t.word_index)
    # print(t.word_docs)
    #
    # # integer encode documents
    # encoded_docs = t.texts_to_matrix(words, mode='count')
    # print(encoded_docs)

    wordsIds = t.texts_to_sequences(words)
    wordnetIds = t.texts_to_sequences(wordnet)

    # # Convert word sentences to ids
    # for sentence in words:
    #     maxsentence = max(maxsentence, len(sentence))
    #     minsentence = min(minsentence, len(sentence))
    #     sentenceId = []
    #
    #     for word in sentence:
    #         if word not in word2id:
    #             word2id[word] = id
    #             id += 1
    #             wordtokens += 1
    #
    #         sentenceId.append(word2id[word])
    #
    #     wordsIds.append(sentenceId)
    #
    # # Convert sense sentences to ids
    # for sentence in wordnet:
    #     sentenceId = []
    #
    #     for word in sentence:
    #         if word not in word2id:
    #             word2id[word] = id
    #             id += 1
    #
    #             if word.startswith('wn:'):
    #                 sensetokens += 1
    #             else:
    #                 print('Error')
    #
    #         sentenceId.append(word2id[word])
    #
    #     wordnetIds.append(sentenceId)

    #wordsIds = pad_sequences(wordsIds, truncating='post', padding='post', maxlen=sentence_size, value=0)
    #wordnetIds = pad_sequences(wordnetIds, truncating='post', padding='post', maxlen=sentence_size, value=0)

    print('\n\nPREPROCESSING SUMMARY\n')
    print('Total word tokens:', wordtokens)
    print('Total sense tokens:', sensetokens)
    print('Biggest sentence:', maxsentence)
    print('Smallest sentence:', minsentence)

    # Print sample sentences
    print('\nSample word sentences')
    for i in words[0:5]:
        print(i)

    print('\nSample wordnet sentences')
    for i in wordnet[0:5]:
        print(i)

    # Print sample sentences with ids
    print('\nSample id sentences')
    for i in wordsIds[0:5]:
        print(i)

    print('\nSample id sense sentences')
    for i in wordnetIds[0:5]:
        print(i)

    return wordsIds, wordnetIds

if __name__ == '__main__':
    args = parse_args()

    _ = load_data(dataset_name=args.dataset_name, path=args.resource_folder, gold_name=args.gold_name)

    # ibis = "long%3:00:02::"
    # lemma =
    #
    # print(synset_id)



    #gold2dic()
    #create_dictionary(args.dataset_name, args.gold_name)