import xml.etree.ElementTree as etree
import pickle
import glob
import re
from argparse import ArgumentParser
from nltk.corpus import wordnet as wn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("train_dataset", nargs='?', default='../resources/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml', help="Name of the xml file to use to train")
    parser.add_argument("dev_dataset", nargs='?', default='../resources/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml', help="Name of the xml file to use to test")
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

def file2dic(filepath, save_file=False):
    mapping = {}

    with open(filepath) as f:
        lines = f.readlines()

        for l in lines:
            map = l.split()
            mapping[map[0]] = map[1]

    if save_file:
        save(mapping, '../resources/gold2dic')

    return mapping

def create_dictionary(dataset_name, gold2dic, train=False):
    words = []
    wordnet = []

    # get an iterable
    context = etree.iterparse(dataset_name)

    sentence = []
    sentenceNet = []
    deletedSentences = 0
    dictionary = {}

    for event, elem in iter(context):

        if elem.tag == "sentence":

            #if(int(elem.attrib['id'])%10 == 0)
            print('\t' + elem.attrib['id'])

            if len(sentence) < 1:
                deletedSentences += 1
            else:
                words.append(' '.join(sentence))
                wordnet.append(' '.join(sentenceNet))
            sentence = []
            sentenceNet = []

        elif elem.tag == "wf" or elem.tag == "instance":
            lemma = elem.attrib["lemma"].lower()
            sentence.append(lemma)

            if elem.tag == "instance":
                dataset_id = elem.attrib["id"]
                synset = wn.lemma_from_key(gold2dic[dataset_id]).synset()
                synset_id = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
                sentenceNet.append(synset_id)
                if lemma not in dictionary:
                    dictionary[lemma] = [synset_id]
                elif synset_id not in dictionary[lemma]:
                    dictionary[lemma].append(synset_id)
            else:
                sentenceNet.append(lemma)

        elem.clear()

    if train:
        save(dictionary, '../resources/' + 'synsetsdic')
        flag = 'train'
    else:
        flag = 'dev'

    save(words, '../resources/' + 'words_' + flag)
    save(wordnet, '../resources/' + 'wordnet_' + flag)

    print('\nSentences removed:', deletedSentences)

    return words, wordnet

def load_data(train_dataset, dev_dataset, path="../resources/", gold_name='gold2dic', sentence_size = 10, num_words=20000):

    # Check if gold2dic exists
    if glob.glob(path + gold_name + '.pkl'):
        print('\nMapping found!')
        gold2dic = load(path + gold_name)
    else:
        print('\nMapping not found!')
        print('\nBuilding mapping from file')
        gold2dic_train = file2dic(filepath= path + "WSD_Evaluation_Framework/Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt")
        gold2dic_dev = file2dic(filepath= path + "WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.key.txt")
        gold2dic = {**gold2dic_train, **gold2dic_dev}
        save(gold2dic, '../resources/gold2dic')

    # Check if train sentences exists
    if glob.glob(path + 'words_train' + '.pkl') and glob.glob(path + 'wordnet_train' + '.pkl'):
        print('\nTrain sentences found!')
        train_words_x = load(path + 'words_train')
        train_words_y = load(path + 'wordnet_train')
    else:
        print('\nTrain sentences not found!')
        train_words_x, train_words_y = create_dictionary(train_dataset, gold2dic, train=True)

    # Check if dev sentences exists
    if glob.glob(path + 'words_dev' + '.pkl') and glob.glob(path + 'wordnet_dev' + '.pkl'):
        print('\nDev sentences found!')
        dev_words_x = load(path + 'words_dev')
        dev_words_y = load(path + 'wordnet_dev')
    else:
        print('\nDev sentences not found!')
        dev_words_x, dev_words_y = create_dictionary(dev_dataset, gold2dic, train=False)

    # Check if tokenizer exists
    if glob.glob(path + 'tokenizer' + '.pkl'):
        print('\nTokenizer found!')
        t = load(path + 'tokenizer')
    else:
        print('\nTokenizer not found!')
        # Tokenizer
        # num_words=20000,
        t = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./;<=>?@[\\]^_`{|}~\'\t', oov_token='<OOV>')
        t.fit_on_texts(train_words_x)
        t.fit_on_texts(train_words_y)
        save(t, path + 'tokenizer')

    # Apply tokenizer
    train_x = t.texts_to_sequences(train_words_x)
    train_y = t.texts_to_sequences(train_words_y)
    dev_x = t.texts_to_sequences(dev_words_x)
    dev_y = t.texts_to_sequences(dev_words_y)

    # Print summary
    print('\nPREPROCESSING SUMMARY')
    print('\tTotal words tokens:', len([y for y in t.word_index if not y.startswith('wn:')]))
    print('\tTotal sense tokens:', len([y for y in t.word_index if y.startswith('wn:')]))
    print('\tBiggest sentence:', max(len(l) for l in train_x))
    print('\tSmallest sentence:', min(len(l) for l in train_x))

    # Print sample sentences
    print('\nSample word sentences')
    for i in train_words_x[0:5]:
        print(i)

    print('\nSample wordnet sentences')
    for i in train_words_y[0:5]:
        print(i)

    # Print sample sentences with ids
    print('\nSample id sentences')
    for i in train_x[0:5]:
        print(i)

    print('\nSample id sense sentences')
    for i in train_y[0:5]:
        print(i)

    # Add padding
    train_x = pad_sequences(train_x, truncating='post', padding='post', maxlen=sentence_size)
    train_y = pad_sequences(train_y, truncating='post', padding='post', maxlen=sentence_size, value=0)
    dev_x = pad_sequences(dev_x, truncating='post', padding='post', maxlen=sentence_size)
    dev_y = pad_sequences(dev_y, truncating='post', padding='post', maxlen=sentence_size, value=0)

    # To categorical
    train_y = to_categorical(train_y, num_classes=num_words)
    dev_y = to_categorical(dev_y, num_classes=num_words)

    print('\nSHAPES')
    print('\tdev_x:', dev_x.shape)
    print('\tdev_y:', dev_y.shape)
    print('\ttrain_x:', train_x.shape)
    print('\ttrain_y:', train_y.shape)

    # Dataset
    dataset = {'train_x': train_x, 'dev_x': dev_x, 'train_y': train_y, 'dev_y': dev_y}

    return dataset, t

if __name__ == '__main__':
    args = parse_args()

    _ = load_data(train_dataset=args.train_dataset, dev_dataset=args.dev_dataset, path=args.resource_folder, gold_name=args.gold_name)
