
import os
import logging
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import json
import numpy as np
from preproces_semEval.parse_text import process_sentence_spacy, parse_sentence_spacy, run_sst

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

def get_sentence_entities(base_dir, train=True):
    print('file didrectory is : ', base_dir)
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

def get_sentence_text(tagged_sentence):
    # print('tagged_sentence Befor xml tree format: ', tagged_sentence)
    tree = ET.XML("<xml>" + tagged_sentence.replace("&", "") + "</xml>")
    # print('tagged_sentence After xml tree format: ', tree)
    sentence_text = ""
    e1_start = 0
    e1 = tree.find("e1")
    e2 = tree.find("e2")
    if tree.text:
        sentence_text = tree.text
        e1_start = len(tree.text)
        # print('tree.text =', tree.text)
    sentence_text += e1.text + e1.tail + e2.text + e2.tail
    print("tree.text ={} e1.text={} e1.tail={}  e2.text={}  e2.tail={}".format(tree.text, e1.text, e1.tail, e2.text,
                                                                               e2.tail))
    return sentence_text, e1, e2


def get_sequences_of_sentence(parsed_sentence_spacy_doc, e1, e2):
    tokens = []
    word_and_POStag_seq = []
    words_posisions_seq = []
    for t in parsed_sentence_spacy_doc:
        tokens.append(t.text.replace(" ", "_").replace('\t', '_').replace('\n', '_'))
        word_and_POStag_seq.append([t.text.replace(" ", "_").replace('\t', '_').replace('\n', '_'), t.tag_])
        # word_posisions_seq .append()
    # sentence_file.write("{}\t{}\t.\n".format(sentence_id, "\t".join(tokens)))
    # get distance(position) of all words rather than to e1 an e2 positions in the sentence

    e1_name = (e1.text).replace(" ", "_").replace('\t', '_').replace('\n', '_')
    e2_name = (e2.text).replace(" ", "_").replace('\t', '_').replace('\n', '_')

    en1_position = 0
    en2_position = 0

    for i in range(len(tokens)):
        if tokens[i] == e1_name:
            en1_position = i
        if tokens[i] == e2_name:
            en2_position = i
    for i in range(len(tokens)):
        # sen_word[s][i] = word2id['BLANK']
        dist_i_to_e1 = pos_embed((i - en1_position), len(tokens))
        dist_i_to_e2 = pos_embed((i - en2_position), len(tokens))
        words_posisions_seq.append([dist_i_to_e1, dist_i_to_e2])
    return tokens, word_and_POStag_seq, words_posisions_seq

def full_sentence_to_json(sentence_id, tagged_text, cleaned_text, sen_word_pos, sentence_sdp_instances,
                          words_dist_to_e1_e2, labels, classe, sentence_entities, sentence_pairs, sentence_wn_instances):

    full_parsed_data = {
        "sentence_id":sentence_id,
        "classe": classe,
        "labels": labels,
        "sentence_entities": sentence_entities,
        "sentence_pairs": str(sentence_pairs),
        "sen_left_sdp": sentence_sdp_instances[0],
        "sen_right_sdp": sentence_sdp_instances[1],
        "sen_tagged_tex": tagged_text,
        "sen_cleaned_text": cleaned_text,
        "sen_word_pos": sen_word_pos,
        "position_to_e1_e2": words_dist_to_e1_e2,
        "left_wordnet": sentence_wn_instances[0],
        "right_wordnet": sentence_wn_instances[1]
    }
    return full_parsed_data

def parse_semeval_sentences(base_dir, out_put_folder, train=True):
    left_sdp_instances = []
    right_sdp_instances = []
    left_wordnet = []
    right_wordnet = []
    left_ancestors = []
    right_wordnet = []
    classes = []
    labels = []
    X_train_word_seq = []
    X_train_position_seq = []
    parsed_sentences = {}
    # first iterate all documents, and preprocess all sentences
    token_seq = {}
    Words_POStag_seq = {}
    sentences_positions_seq = {}
    max_seq_len = 0
    sentences_without_sdp = [] #TODO VAHAB
    entities = get_sentence_entities(base_dir, train=True)
    with open(base_dir[0], 'r') as f:
        data = f.read()
    if train:
        lines = data.split("\n\n")
    else:
        lines = data.split("\n")
    f.close()
    del data
    json_big_file = open(out_put_folder+"_parsed_All_sentences.json", 'a')

    for line in lines:
        if line.strip():
            if train:
                tagged_sentence, sentence_label, comment = line.split("\n")
            else:
                tagged_sentence = line
            sentence_id, tagged_sentence = tagged_sentence.split("\t")
            tagged_sentence = tagged_sentence[1:-1] # to remove '"' mark from begin and end of the sentence
            sentence_pairs = {}
            if train and sentence_label != "Other":
                sentence_pairs[("{}e1".format(sentence_id), "{}e2".format(sentence_id))] = label_to_pairtype[sentence_label]
            sentence_entities = entities[sentence_id]
            sentence_text, e1, e2 = get_sentence_text(tagged_sentence)
            parsed_sentence_spacy_doc = parse_sentence_spacy(sentence_text, sentence_entities)
            print("spacy_parsed_sentences.text= ", parsed_sentence_spacy_doc)
            tokens, word_and_POStag_seq, words_positions_seq = get_sequences_of_sentence(parsed_sentence_spacy_doc,e1,e2)
            #update max_seq_len
            if len(tokens) > max_seq_len:
                max_seq_len = len(tokens)
            wordnet_sentence = wordnet_tags = run_sst(token_seq)
            # parsed_sentences[sentence_id] = parsed_sentence_spacy_doc
            # token_seq[sentence_id] = tokens
            # Words_POStag_seq[sentence_id]= word_and_POStag_seq
            # sentences_positions_seq[sentence_id] = words_posisions_seq
            if sentence_id == 5 or sentence_id == '5':
                print("sentence without interween sdp")
            sentence_labels, sentence_sdp_instances, sentence_wn_instances, sentence_classes,_,_ = \
                process_sentence_spacy(parsed_sentence_spacy_doc, sentence_entities, sentence_pairs, wordnet_sentence)

            # write parsed data to a json file
            full_sentence_in_json = full_sentence_to_json(sentence_id, tagged_sentence,sentence_text, word_and_POStag_seq,
                                                          sentence_sdp_instances, words_positions_seq, sentence_labels,
                                                          sentence_classes,sentence_entities, sentence_pairs,
                                                          sentence_wn_instances)

            full_sentence_json = json.dumps(full_sentence_in_json)
            json_big_file.write(full_sentence_json)
            json_big_file.write("\n")

            if len(sentence_sdp_instances[0])> 0:
                labels += sentence_labels
                X_train_word_seq.append(word_and_POStag_seq)
                X_train_position_seq.append(words_positions_seq)
                if len(word_and_POStag_seq)!= len(words_positions_seq):
                    print(".. missing words in sequencing process in sentnece -->", sentence_id)
                left_sdp_instances += sentence_sdp_instances[0]
                right_sdp_instances += sentence_sdp_instances[1]
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
    print('sem_eval_data_max_seq_len=', max_seq_len)
    print('sentences_without_sdp=',sentences_without_sdp)
    print('len(sentences_without_sdp=)', len(sentences_without_sdp))

    json_big_file.close()
    with open(out_put_folder + "_sentences_without_sdp_id.json", 'w') as sentences_without_sdp_json_file :
        json.dump(sentences_without_sdp,sentences_without_sdp_json_file)
    return labels, X_train_word_seq, X_train_position_seq, (left_sdp_instances,right_sdp_instances),classes,(left_wordnet,right_wordnet)

def prepair_train_data(input_text_file, out_put_folder="../temp/semeval8train/semeval8train"):
    """
    :param input_text_file: training raw text file of semeval_2010 task 8 data
    :param out_put_folder: temp folder of preproccesd train data
    :return:
    """
    argv = ["","","",""]
    argv[3] = out_put_folder
    labels, X_train_word_pos_seq, X_train_positions_seq, sdp_instances, classes, wordnet = \
        parse_semeval_sentences([input_text_file],out_put_folder)
    print('labels=', labels)
    print("classes= ", classes)
    print('instances = ', sdp_instances[0][0:len(sdp_instances[0])])
    print('\nX_train_word_pos_seq[0:5] =\n', X_train_word_pos_seq[0:5])
    print('X_train_positions_seq[0:5] =\n', X_train_positions_seq[0:5])
    print('len(labels)=', len(labels))
    print('len(X_train_word_pos_seq)=', len(X_train_word_pos_seq))
    print('len(X_train_positions_seq)=', len(X_train_positions_seq))
    Y_train = classes
    train_labels = labels
    X_sdp_train = sdp_instances
    X_train_wordnet = wordnet
    # ba in Component - Whole(e2, e1) e2-21 chikar mikoni?
    np.save(argv[3] + "_x_word_pos_seq.npy", X_train_word_pos_seq)
    np.save(argv[3] + "_x_positions_seq.npy", X_train_positions_seq)
    np.save(argv[3] + "_labels.npy", train_labels)
    np.save(argv[3] + "_x_sdp_words.npy", X_sdp_train)
    np.save(argv[3] + "_y.npy", Y_train)
    np.save(argv[3] + "_x_wordnet.npy", X_train_wordnet)

def prepair_test_data(input_text_file, out_put_folder="../temp/semeval8test/semeval8test"):
    """
    :param input_text_file: training raw text file of semeval_2010 task 8 data
    :param out_put_folder: temp folder of preproccesd test data
    :return:
    """
    argv = ["","","",""]
    argv[3] = out_put_folder
    labels, X_train_word_pos_seq, X_train_positions_seq, sdp_instances, classes, wordnet = \
        parse_semeval_sentences([input_text_file],out_put_folder)
    print('labels=', labels)
    print("classes= ", classes)
    print('instances = ', sdp_instances[0][0:len(sdp_instances[0])])
    print('\nX_train_word_pos_seq[0:5] =\n', X_train_word_pos_seq[0:5])
    print('X_train_positions_seq[0:5] =\n', X_train_positions_seq[0:5])
    print('len(labels)=', len(labels))
    print('len(X_train_word_pos_seq)=', len(X_train_word_pos_seq))
    print('len(X_train_positions_seq)=', len(X_train_positions_seq))
    Y_train = classes
    train_labels = labels
    X_sdp_train = sdp_instances
    X_train_wordnet = wordnet
    # ba in Component - Whole(e2, e1) e2-21 chikar mikoni?
    np.save(argv[3] + "_x_word_pos_seq.npy", X_train_word_pos_seq)
    np.save(argv[3] + "_x_positions_seq.npy", X_train_positions_seq)
    np.save(argv[3] + "_labels.npy", train_labels)
    np.save(argv[3] + "_x_sdp_words.npy", X_sdp_train)
    np.save(argv[3] + "_y.npy", Y_train)
    np.save(argv[3] + "_x_wordnet.npy", X_train_wordnet)


def load_semEvalt8_train_data(input_data_path="../temp/semeval8train/semeval8train", channels=["word_seq", "positions_seq","sdp_words" ]):
    argv = ["", "", "", "", ""]
    argv[2] = input_data_path
    # is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi("{}/chebi.obo".format(DATA_DIR))

    X_word_seq_train = None
    X_position_seq_train = None
    X_sdp_words_train = None
    X_wordnet_train = None
    X_subpaths_train = None
    X_ancestors_train = None

    train_labels = np.load(argv[2] + "_labels.npy")
    # is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()  # "{}/chebi.obo".format(DATA_DIR)TODO VAHAB added here
    id_to_index = []
    Y_train = np.load(argv[2] + "_y.npy")
    # Y_train = to_categorical(Y_train, num_classes=n_classes)

    if "word_seq" in channels:
        X_word_seq_train = np.load(argv[2] + "_x_word_pos_seq.npy")
        print("....", len(X_word_seq_train))
    if "positions_seq" in channels:
        X_position_seq_train = np.load(argv[2] + "_x_positions_seq.npy")
    if "sdp_words" in channels:
        X_sdp_words_train = np.load(argv[2] + "_x_sdp_words.npy")  # haman sdp hastesh ke jaygozin kol matn shodeh ast ....
    if "wordnet" in argv[4:]:
        X_wordnet_train = np.load(argv[2] + "_x_wordnet.npy")
    if "concat_ancestors" in argv[4:] or "common_ancestors" in argv[4:]:
        X_subpaths_train = np.load(argv[2] + "_x_subpaths.npy")
        X_ancestors_train = np.load(argv[2] + "_x_ancestors.npy")

    print('X_sdp_words_train.shape = ', X_sdp_words_train.shape)
    print('X_sdp_words_train[0] = ', X_sdp_words_train[0])

    return train_labels, Y_train, X_word_seq_train, X_position_seq_train, X_sdp_words_train, X_wordnet_train

def load_semEvalt8_test_data(input_data_path, channels=["word_seq", "positions_seq","sdp_words" ]):
    argv = ["", "", "", "", ""]
    argv[2] = input_data_path
    X_word_seq_train = None
    X_position_seq_train = None
    X_sdp_words_train = None
    X_wordnet_train = None
    X_subpaths_train = None
    X_ancestors_train = None
    # is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi("{}/chebi.obo".format(DATA_DIR))
    train_labels = np.load(argv[2] + "_labels.npy")
    # is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = load_chebi()  # "{}/chebi.obo".format(DATA_DIR)TODO VAHAB added here
    id_to_index = []
    Y_train = np.load(argv[2] + "_y.npy")
    # Y_train = to_categorical(Y_train, num_classes=n_classes)
    if "word_seq" in channels:
        X_word_seq_train = np.load(argv[2] + "_x_word_pos_seq.npy")
        print("....", len(X_word_seq_train))
    if "positions_seq" in channels:
        X_position_seq_train = np.load(argv[2] + "_x_positions_seq.npy")
    if "sdp_words" in channels:
        X_sdp_words_train = np.load(
            argv[2] + "_x_sdp_words.npy")  # haman sdp hastesh ke jaygozin kol matn shodeh ast ....
    if "wordnet" in argv[4:]:
        X_wordnet_train = np.load(argv[2] + "_x_wordnet.npy")
    if "concat_ancestors" in argv[4:] or "common_ancestors" in argv[4:]:
        X_subpaths_train = np.load(argv[2] + "_x_subpaths.npy")
        X_ancestors_train = np.load(argv[2] + "_x_ancestors.npy")

    print('X_sdp_words_train.shape = ', X_sdp_words_train.shape)
    print('X_sdp_words_train[0] = ', X_sdp_words_train[0])
    return train_labels, Y_train, X_word_seq_train, X_position_seq_train, X_sdp_words_train, X_wordnet_train


if __name__ == "__main__":
    #get_semeval8_sdp_instances("../data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT")

    train_text_file = "../data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    test_text_file = "../data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

    prepair_train_data(input_text_file=train_text_file, out_put_folder="../temp/semeval8train/semeval8train")
    prepair_test_data(input_text_file=test_text_file, out_put_folder="../temp/semeval8test/semeval8test")
""""
    channels = ["sdp_words", "word_seq", "positions_seq","pos_tag"]
    training_data_path = "../temp/semeval8train/semeval8train"
    test_data_path = "../temp/semeval8test/semeval8test"
    X_word_seq_train = None
    X_position_seq_train = None
    X_sdp_words_train = None
    X_wordnet_train = None
    X_subpaths_train = None
    X_ancestors_train = None
    train_labels, Y_train, X_word_seq_train, X_position_seq_train, X_sdp_words_train, X_wordnet_train = \
        load_semEvalt8_train_data(input_data_path=training_data_path, channels=channels)

    train_labels, Y_test, X_word_seq_test, X_position_seq_test, X_sdp_words_ttest, X_wordnet_ttest = \
        load_semEvalt8_test_data(input_data_path=test_data_path, channels=channels)

"""