import xml.etree.ElementTree as etree
import pickle
import glob
import re
from argparse import ArgumentParser
from nltk.corpus import wordnet as wn

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

    for event, elem in iter(context):

        if elem.tag == "sentence":# and event == 'start':
            print(elem.attrib['id'])
            words.append(sentence)
            wordnet.append(sentenceNet)
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

    return words, wordnet

def load_data(dataset_name, path="../resources/", gold_name='gold2dic'):

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
        words = load(path + 'wordnet')
    else:
        words, wordnet = create_dictionary(dataset_name, gold2dic)

    # Print sample sentences
    print('\nSample word sentences')
    for i in words[0:10]:
        print(i)

    print('\nSample wordnet sentences')
    for i in wordnet[0:10]:
        print(i)

    return words, wordnet

if __name__ == '__main__':
    args = parse_args()

    _ = load_data(dataset_name=args.dataset_name, path=args.resource_folder, gold_name=args.gold_name)

    # ibis = "long%3:00:02::"
    # lemma =
    #
    # print(synset_id)



    #gold2dic()
    #create_dictionary(args.dataset_name, args.gold_name)