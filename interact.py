from typing import List

from model import CrossModel
from dataset import dataset, CRSdataset
import torch
from torch import optim
import torch.nn as nn
import pickle as pkl
import argparse
import json
import signal
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
from nltk import word_tokenize
from functools import reduce
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# user package


def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_c_length", "--max_c_length", type=int, default=256)
    train.add_argument("-max_r_length", "--max_r_length", type=int, default=30)
    train.add_argument("-batch_size", "--batch_size", type=int, default=32)
    train.add_argument("-max_count", "--max_count", type=int, default=5)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-load_dict", "--load_dict", type=str, default=None)
    train.add_argument("-learningrate", "--learningrate", type=float, default=1e-3)
    train.add_argument("-optimizer", "--optimizer", type=str, default='adam')
    train.add_argument("-momentum", "--momentum", type=float, default=0)
    train.add_argument("-is_finetune", "--is_finetune", type=bool, default=False)
    train.add_argument("-embedding_type", "--embedding_type", type=str, default='random')
    train.add_argument("-epoch", "--epoch", type=int, default=30)
    train.add_argument("-gpu", "--gpu", type=str, default='0,1')
    train.add_argument("-gradient_clip", "--gradient_clip", type=float, default=0.1)
    train.add_argument("-embedding_size", "--embedding_size", type=int, default=300)

    train.add_argument("-n_heads", "--n_heads", type=int, default=2)
    train.add_argument("-n_layers", "--n_layers", type=int, default=2)
    train.add_argument("-ffn_size", "--ffn_size", type=int, default=300)

    train.add_argument("-dropout", "--dropout", type=float, default=0.1)
    train.add_argument("-attention_dropout", "--attention_dropout", type=float, default=0.0)
    train.add_argument("-relu_dropout", "--relu_dropout", type=float, default=0.1)

    train.add_argument("-learn_positional_embeddings",
                       "--learn_positional_embeddings", type=bool, default=False)
    train.add_argument("-embeddings_scale", "--embeddings_scale", type=bool, default=True)

    train.add_argument("-n_entity", "--n_entity", type=int, default=64368)
    train.add_argument("-n_relation", "--n_relation", type=int, default=214)
    train.add_argument("-n_concept", "--n_concept", type=int, default=29308)
    train.add_argument("-n_con_relation", "--n_con_relation", type=int, default=48)
    train.add_argument("-dim", "--dim", type=int, default=128)
    train.add_argument("-n_hop", "--n_hop", type=int, default=2)
    train.add_argument("-kge_weight", "--kge_weight", type=float, default=1)
    train.add_argument("-l2_weight", "--l2_weight", type=float, default=2.5e-6)
    train.add_argument("-n_memory", "--n_memory", type=float, default=32)
    train.add_argument("-item_update_mode", "--item_update_mode", type=str, default='0,1')
    train.add_argument("-using_all_hops", "--using_all_hops", type=bool, default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)

    return train


class IConversationalRecommender():
    def __init__(self, opt):
        self.opt = opt
        self.model = CrossModel(self.opt, is_finetune=False)

        self.logs = []
        self.context = []

    def prompt(self):
        self.input = input('KGSF> ')

        # TODO add popup of movie items for selection


class Pocessor():
    def __init__(self, opt):
        # dbpedia (kg of movie)
        self.id2entity = pkl.load(open('data/id2entity.pkl', 'rb'))
        self.entity2entityId = pkl.load(open('data/entity2entityId.pkl', 'rb'))
        self.entity_max = len(self.entity2entityId)

        # concepts from conceptNet
        self.key2index = json.load(open('data/key2index_3rd.json', encoding='utf-8'))

        # word from corpus trained by gensim
        self.word2index = json.load(open('data/word2index_redial.json', encoding='utf-8'))

        # parameter
        self.max_c_length = opt['max_c_length']
        self.max_count = opt['max_count']
        self.entity_num = opt['n_entity']

    def data_process(self, log: List[str]):
        # tokenize context
        contexts = [self.tokenize(sen) for sen in log]
        # extract movie from context
        entities = reduce(lambda x, y: x+y, [self.detect_movie(sen) for sen in contexts])
        entities_dedup = set()
        for entity in entities:
            entities_dedup.add(entity)
        entities_dedup = list(entities_dedup)

        # index context and get mask of concept net and dbpedia
        indexed_context, c_lengths, concept_mask, dbpedia_mask = \
            self.padding_context(contexts)
        return indexed_context, c_lengths, concept_mask, dbpedia_mask, entities_dedup

    def tokenize(self, sentence: str) -> List[str]:
        token_text = word_tokenize(sentence)
        num = 0
        token_text_ret = []

        while num < len(token_text):
            if token_text[num] == '@' and num != len(token_text)-1:
                token_text_ret.append(token_text[num]+token_text[num+1])
                num += 2
            else:
                token_text_ret.append(token_text[num])
                num += 1

        return token_text_ret

    def detect_movie(self, tokenize_sen: List[str]):
        movie_trans = []
        for word in tokenize_sen:
            if word[0] == '@':
                try:
                    entity = self.id2entity[int(word[1:])]
                    movie_trans.append(self.entity2entityId[entity])
                except:
                    pass
        return movie_trans

    def padding_context(self, contexts: List[List[str]], pad=0):
        contexts_ret = []

        for sen in contexts[-self.max_count:-1]:
            contexts_ret.extend(sen)
            contexts_ret.append('_split_')
        contexts_ret.extend(contexts[-1])

        indexed_context, c_lengths, concept_mask, dbpedia_mask = \
            self.padding_w2v(contexts_ret, self.max_c_length)
        return indexed_context, c_lengths, concept_mask, dbpedia_mask

    def padding_w2v(self, sentence: List[str], max_length: int, pad=0, end=2, unk=3):
        sen_vector, concept_mask, dbpedia_mask = [], [], []

        for word in sentence:
            # sentence vector
            sen_vector.append(self.word2index.get(word, unk))
            # concept vector
            concept_mask.append(self.key2index.get(word.lower(), 0))
            # movie vector
            if '@' in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    entityId = self.entity2entityId[entity]
                except:
                    entityId = self.entity_max
                dbpedia_mask.append(entityId)
            else:
                dbpedia_mask.append(self.entity_max)

        # end of sentence
        sen_vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)

        # pad or truncate
        if len(sen_vector) > max_length:
            return sen_vector[:max_length], \
                max_length, \
                concept_mask[:max_length], \
                dbpedia_mask[:max_length]
        else:
            length = len(sen_vector)
            return sen_vector+(max_length-length)*[pad], \
                length,\
                concept_mask+(max_length-length)*[0], \
                dbpedia_mask+(max_length-length)*[self.entity_max]


if __name__ == '__main__':
    opt = setup_args().parse_args()

    processor = Pocessor(vars(opt))

    while True:
        sentence = input('KGSF> ')

        if sentence != '':
            vector, length, concept_mask, dbpedia_mask, entities = processor.data_process(
                sentence)
            print(length, f'entities: {entities}',
                  f'vector: {vector[:length]}',
                  f'concept_mask: {concept_mask[:length]}',
                  f'dbpedia_mask: {dbpedia_mask[:length]}',
                  sep='\n')
        else:
            break
