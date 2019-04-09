import os
import logging
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import json
from src.parse_text import process_sentence_spacy, parse_sentence_spacy, run_sst

OTHER = 0
REL1 = 1
REL2 = 2
REL3 = 3
REL4 = 4
REL5 = 5
REL6 = 6
REL7 = 7
REL8 = 8
REL9 = 9
REL1I = 10
REL2I = 11
REL3I = 12
REL4I = 13
REL5I = 14
REL6I = 15
REL7I = 16
REL8I = 17
REL9I = 18

pairtypes = (REL1, REL2, REL3, REL4, REL5, REL6, REL7, REL8, REL9,
             REL1I, REL2I, REL3I, REL4I, REL5I, REL6I, REL7I, REL8I, REL9I)
label_to_pairtype = {"Cause-Effect(e1,e2)": REL1,
                     "Instrument-Agency(e1,e2)": REL2,
                     "Product-Producer(e1,e2)": REL3,
                     "Content-Container(e1,e2)": REL4,
                     "Entity-Origin(e1,e2)": REL5,
                     "Entity-Destination(e1,e2)": REL6,
                     "Component-Whole(e1,e2)": REL7,
                     "Member-Collection(e1,e2)": REL8,
                     "Message-Topic(e1,e2)": REL9,
                     "Cause-Effect(e2,e1)": REL1I,
                     "Instrument-Agency(e2,e1)": REL2I,
                     "Product-Producer(e2,e1)": REL3I,
                     "Content-Container(e2,e1)": REL4I,
                     "Entity-Origin(e2,e1)": REL5I,
                     "Entity-Destination(e2,e1)": REL6I,
                     "Component-Whole(e2,e1)": REL7I,
                     "Member-Collection(e2,e1)": REL8I,
                     "Message-Topic(e2,e1)": REL9I
                     }

n_classes = 19

#length of sentence
fixlen = 120
#max length of position embedding is 100 (-100~+100)
maxlen = 100

def get_sentence_entities(base_dir, name_to_id, synonym_to_id, train):
    entities = {}  # sentence_id -> entities
    with open(base_dir[0], 'r') as f:
        data = f.read()
    if train:
        lines = data.split("\n\n")
    else:
        lines = data.split("\n")
    f.close()
    del data
    for line in lines:
        if line.strip():
            if train:
                tagged_sentence, sentence_label, comment = line.split("\n")
            else:
                tagged_sentence = line
            sentence_id, tagged_sentence = tagged_sentence.split("\t")
            tagged_sentence = tagged_sentence[1:-1]
            #print(tagged_sentence)
            tree = ET.XML("<xml>" + tagged_sentence.replace("&", "") + "</xml>")
            #root = tree.getroot()
            #sentence_text = ET.tostring(tree, encoding='utf8', method='text')
            e1 = tree.find("e1")
            e2 = tree.find("e2")
            e1_start = 0
            if tree.text:
                e1_start = len(tree.text)
            e1_end = e1_start + len(e1.text)
            e2_start = e1_end + len(e1.tail)
            e2_end = e2_start + len(e2.text)
            entities[sentence_id] = {"{}e1".format(sentence_id): ((e1_start, e1_end), e1.text, ""),
                                     "{}e2".format(sentence_id): ((e2_start, e2_end), e2.text, "")}
    return entities

def pos_embed(x,sentence_len): # changing negetive distances to a positives numbers
    return max(0, min(x + maxlen, maxlen + maxlen + 1))
# def pos_embed(x, sentence_len): # changing negetive distances to a positives numbers
#     return max(0, min(x + sentence_len, sentence_len + sentence_len + 1))

def parse_semeval_sentences(base_dir, entities, train):
    parsed_sentences = {}
    # first iterate all documents, and preprocess all sentences
    token_seq = {}
    Words_POStag_seq = {}
    sentences_positions_seq = {}

    max_seq_len = 0

    with open(base_dir[0], 'r') as f:
        data = f.read()
    if train:
        lines = data.split("\n\n")
    else:
        lines = data.split("\n")

    for line in lines:
        if line.strip():
            if train:
                tagged_sentence, sentence_label, comment = line.split("\n")
            else:
                tagged_sentence = line
            sentence_id, tagged_sentence = tagged_sentence.split("\t")
            tagged_sentence = tagged_sentence[1:-1]
            #print('tagged_sentence Befor xml tree format: ', tagged_sentence)
            tree = ET.XML("<xml>" + tagged_sentence.replace("&", "") + "</xml>")
            #print('tagged_sentence After xml tree format: ', tree)
            sentence_entities =  entities[sentence_id]
            sentence_text = ""
            e1_start = 0
            e1 = tree.find("e1")
            e2 = tree.find("e2")
            if tree.text:
                sentence_text = tree.text
                e1_start = len(tree.text)
                # print('tree.text =', tree.text)
            sentence_text += e1.text + e1.tail + e2.text + e2.tail
            print("tree.text ={} e1.text={} e1.tail={}  e2.text={}  e2.tail={}".format(tree.text,e1.text, e1.tail, e2.text, e2.tail))
            parsed_sentence_spacy_doc = parse_sentence_spacy(sentence_text, sentence_entities)
            parsed_sentences[sentence_id] = parsed_sentence_spacy_doc
            print("spacy_parsed_sentences.text= ", parsed_sentence_spacy_doc)


            tokens = []
            word_and_POStag_seq = []
            # words_posisions_seq = []
            for t in parsed_sentence_spacy_doc:
                tokens.append(t.text.replace(" ", "_").replace('\t', '_').replace('\n', '_'))
                word_and_POStag_seq.append([t.text.replace(" ", "_").replace('\t', '_').replace('\n', '_'), t.tag_ ])
                #word_posisions_seq .append()
            # sentence_file.write("{}\t{}\t.\n".format(sentence_id, "\t".join(tokens)))
            #get distance(position) of all words rather than to e1 an e2 positions in the sentence

            e1_name = (e1.text).replace(" ", "_").replace('\t', '_').replace('\n', '_')
            e2_name = (e2.text).replace(" ", "_").replace('\t', '_').replace('\n', '_')

            en1_position = 0
            en2_position = 0
            words_posisions_seq = []

            if len(tokens)>max_seq_len:
                max_seq_len = len(tokens)

            for i in range(len(tokens)):
                if tokens[i] == e1_name:
                    en1_position = i
                if tokens[i] == e2_name:
                    en2_position = i
            for i in range(len(tokens)):
                # sen_word[s][i] = word2id['BLANK']
                dist_i_to_e1 = pos_embed((i - en1_position),len(tokens))
                dist_i_to_e2 = pos_embed((i - en2_position),len(tokens))
                words_posisions_seq.append([dist_i_to_e1,dist_i_to_e2])

            token_seq[sentence_id] = tokens
            Words_POStag_seq[sentence_id]= word_and_POStag_seq
            sentences_positions_seq[sentence_id] = words_posisions_seq
    wordnet_tags = run_sst(token_seq)
    return parsed_sentences,Words_POStag_seq, sentences_positions_seq, wordnet_tags


def get_semeval8_sdp_instances(base_dir, train=True):
    """
    Parse DDI corpus, return vectors of SDP of each relation instance
    :param base_dir: directory containing semeval XML documents and annotations
    :return: instances (vectors), classes (0/1) and labels (eid1, eid2)
    """
    left_instances = []
    right_instances = []
    left_wordnet = []
    right_wordnet = []
    left_ancestors = []
    right_wordnet = []
    classes = []
    labels = []
    X_train_word_seq = []
    X_train_position_seq = []

    wordnet_sentence = [] #TODO VAHAB
    sentences_without_sdp = [] #TODO VAHAB

    entities = get_sentence_entities(base_dir, None, None, train)
    parsed_sentences,words_postag_sentences, sentences_positions_seq, wordnet_sentences = parse_semeval_sentences(base_dir, entities, train)

    with open(base_dir[0], 'r') as f:
        data = f.read()
    if train:
        lines = data.split("\n\n")
    else:
        lines = data.split("\n")

    f.close()
    del data

    for line in lines:
        if line.strip():
            if train:
                tagged_sentence, sentence_label, comment = line.split("\n")
            else:
                tagged_sentence = line
            sentence_id, tagged_sentence = tagged_sentence.split("\t")
            sentence_pairs = {}
            if train and sentence_label != "Other":
                sentence_pairs[("{}e1".format(sentence_id), "{}e2".format(sentence_id))] = label_to_pairtype[sentence_label]
            # print(sentence_entities, sentence_pairs)
            #sentence_pairs = {(p.get("e1"), p.get("e2")): label_to_pairtype[p.get("type")] for p in sentence.findall('pair') if p.get("ddi") == "true"}
            # sentence_pairs: {(e1id, e2id): pairtype_label}
            sentence_entities = entities[sentence_id]
            parsed_sentence = parsed_sentences[sentence_id]

            # word and pos_tag informations like [[w1, postag1][w2,postag2][wordtag,postag3],[,], [,],...]
            tokenized_sentence = words_postag_sentences[sentence_id]
            # distantances of each words rather than e1 and e2 positions like, [[0,10],[1,9][2,8][3,7],...]
            position_seqs_sentence = sentences_positions_seq[sentence_id]
            try:
                wordnet_sentence = wordnet_sentences[sentence_id]
            except:
                print('empty wordnet result')
            # sentence_pairs: {(e1id, e2id): pairtype_label}

            sentence_labels, sentence_we_instances, sentence_wn_instances, sentence_classes,_,_ = \
                process_sentence_spacy(parsed_sentence, sentence_entities, sentence_pairs, wordnet_sentence)

            if len(sentence_we_instances[0])> 0:
                labels += sentence_labels

                X_train_word_seq.append(tokenized_sentence)
                X_train_position_seq.append(position_seqs_sentence)

                left_instances += sentence_we_instances[0]
                right_instances += sentence_we_instances[1]

                left_wordnet += sentence_wn_instances[0]
                right_wordnet += sentence_wn_instances[1]
                classes += sentence_classes
                print("++++++ classes=", classes, "\nlen(classes)=", len(classes))
                print("++++++ labels=", labels,"\nlen(labels)= ", len(labels))

                # TODO 2 jomleh paein yek list az koll dadeh ha misakht ama man listi az sequence haye jomleh ha mikhastam
                # "+=" --> listi az kol dadehha ba jomalate fllaten

                # X_train_word_seq += tokenized_sentence
                # X_train_position_seq += position_seqs_sentence

                # ".append" -->  listi  az sequence haye jomleh ha
            else:
                sentences_without_sdp.append(sentence_id)
    with open('failed_sdp_sentences_index_semeval2010_task8.json', 'w',encoding="utf8") as jsonfile:
        json.dump(sentences_without_sdp, jsonfile)
    return labels, X_train_word_seq, X_train_position_seq, (left_instances, right_instances), classes, (None, None), (left_wordnet, right_wordnet)

if __name__ == "__main__":
    #get_semeval8_sdp_instances("../data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT")
    labels,X_train_word_pos_seq, X_train_position_seq, instances,classes, _, wordnet= \
        get_semeval8_sdp_instances(["../data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"])
    print('labels[0]=',labels[0])
    print("classes[0]= ",classes[0])
    print('instances = ', instances[0][0:len(instances[0])])
    print('\nX_train_word_pos_seq =\n',X_train_word_pos_seq[0:150])
    print('X_train_position_seq =\n',X_train_position_seq[0:150])

