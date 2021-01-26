from typing import List

from model import CrossModel, TrainType
from dataset import dataset, CRSdataset
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn as nn
import pickle as pkl
import argparse
import json
import numpy as np
from tqdm import tqdm
import os
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

    # model parameter
    train.add_argument("-model_path", "--model_path", type=str,
                       default='saved_model/net_parameter1.pkl')

    return train


class Processor():
    def __init__(self, opt):
        # dbpedia (kg of movie)
        self.id2entity = pkl.load(open('data/id2entity.pkl', 'rb'))
        self.entity2entityId = pkl.load(open('data/entity2entityId.pkl', 'rb'))
        self.entity_pad = len(self.entity2entityId)
        self.concept_pad = 0

        # concepts from conceptNet (kg of concepts)
        self.key2index = json.load(open('data/key2index_3rd.json', encoding='utf-8'))

        # word from corpus trained by gensim
        self.word2index = json.load(open('data/word2index_redial.json', encoding='utf-8'))

        # parameter
        self.max_c_length = opt['max_c_length']
        self.max_count = opt['max_count']
        self.entity_num = opt['n_entity']
        self.concept_num = opt['n_concept'] + 1

    def data_process(self, logs: List[str]):
        # tokenize context
        contexts = [self.tokenize(sen) for sen in logs]

        # extract movie from context
        entities = reduce(lambda x, y: x+y, [self.detect_movie(sen) for sen in contexts])
        entities_dedup = set()
        for en in entities:
            entities_dedup.add(en)
        entities_dedup = np.array(list(entities_dedup), dtype=np.int64)

        # index context and get mask of concept net and dbpedia
        indexed_context, c_lengths, concept_mask, dbpedia_mask = \
            self.padding_context(contexts)

        # to ndarray
        indexed_context = np.array(indexed_context)
        concept_mask = np.array(concept_mask)
        dbpedia_mask = np.array(dbpedia_mask)

        # bit mask
        entity_bitmask = np.zeros(self.entity_num)
        db_bitmask = np.zeros(self.entity_num)
        concept_bitmask = np.zeros(self.concept_num)

        if len(entities_dedup) != 0:
            entity_bitmask[entities_dedup] = 1
        concept_bitmask[concept_mask] = 1
        db_bitmask[dbpedia_mask] = 1

        # modification
        entities_dedup = np.pad(entities_dedup, (0, 50-len(entities_dedup)), 'constant', constant_values=(0, 0)) \
            if 50 > len(entities_dedup) else entities_dedup
        concept_bitmask[self.concept_pad] = db_bitmask[self.entity_pad] = 0

        return indexed_context, c_lengths, concept_mask, concept_bitmask, \
            dbpedia_mask, db_bitmask, entities_dedup, entity_bitmask

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
            concept_mask.append(self.key2index.get(word.lower(), self.concept_pad))
            # movie vector
            if '@' in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    entityId = self.entity2entityId[entity]
                except:
                    entityId = self.entity_pad
                dbpedia_mask.append(entityId)
            else:
                dbpedia_mask.append(self.entity_pad)

        # end of sentence
        sen_vector.append(end)
        concept_mask.append(self.concept_pad)
        dbpedia_mask.append(self.entity_pad)

        # pad or truncate
        if len(sen_vector) > max_length:
            return sen_vector[:max_length], \
                np.array(max_length), \
                concept_mask[:max_length], \
                dbpedia_mask[:max_length]
        else:
            length = np.array(len(sen_vector))
            return sen_vector+(max_length-length)*[pad], \
                length,\
                concept_mask+(max_length-length)*[self.concept_pad], \
                dbpedia_mask+(max_length-length)*[self.entity_pad]


class IConversationalRecommender():
    def __init__(self, opt):
        # tensorboard
        self.writer = SummaryWriter()

        # word from corpus trained by gensim
        self.word2index = json.load(open('data/word2index_redial.json', encoding='utf-8'))
        self.index2word = {self.word2index[key]: key for key in self.word2index}

        # mapping from movie ids to names
        self.id2moviename = pkl.load(open('data/movie_id2name.pkl', 'rb'))

        # model
        self.opt = opt
        self.model = CrossModel(opt, self.word2index, is_finetune=True).cuda()
        self.processor = Processor(opt)

        # conversation logs
        self.logs = []

    def visualize_model(self):
        sample_sen = 'Sample sentence for model visualization'
        context, length, concept_mask, concept_bitmask, dbpedia_mask, dbpedia_bitmask, \
            entities, entity_bitmask = self.to_batch_tensor(
                *(self.processor.data_process([sample_sen])))

        seed_sets = [123]
        self.writer.add_graph(self.model, (context.cuda(), concept_mask, dbpedia_mask,
                                           concept_bitmask, dbpedia_bitmask, seed_sets,
                                           entities.cuda(), 3))

    def prompt(self):
        self.input = input('KGSF> ')

        self.logs.append(self.input)

        # TODO add popup of movie items for selection

    def to_batch_tensor(self, *args):
        args = [torch.from_numpy(arg) for arg in args]
        return [torch.unsqueeze(arg, 0) for arg in args]

    def vector2sentence(self, sen: List[int]):
        sentence_id = []
        sentence_name = []
        for idx in sen:
            try:
                if idx >= 3:
                    # without replace movie id with name
                    word = '_UNK_' if idx == 3 else self.index2word[idx]
                    sentence_id.append(word)
                    # replace id with name
                    word = self.convert_id_to_name(word) if word[0] == '@' else word
                    sentence_name.append(word)
            except:
                print("OOV", idx)
        return ' '.join(w for w in sentence_name), ' '.join(w for w in sentence_id)

    def convert_id_to_name(self, movieId: str) -> str:
        try:
            return self.id2moviename[movieId[1:]]
        except:
            return movieId

    def start(self):
        self.model.load_model(model_path=self.opt['model_path'])
        while True:
            self.prompt()

            if self.input == '':
                print('End of conversation ...')
                self.logs = []
                continue
            elif self.input == 'exit()':
                break

            # get model input from logs
            context, length, concept_mask, concept_bitmask, dbpedia_mask, dbpedia_bitmask, \
                entities, entity_bitmask = self.processor.data_process(self.logs)
            seed_sets = [entities.nonzero()[0].tolist()]

            # inference
            self.model.eval()
            with torch.no_grad():
                context, concept_mask, concept_bitmask, dbpedia_mask, dbpedia_bitmask, entities, entity_bitmask = \
                    self.to_batch_tensor(context, concept_mask, concept_bitmask,
                                         dbpedia_mask, dbpedia_bitmask, entities, entity_bitmask)

                scores, preds, rec_scores, _, _, _, _, _ = self.model(
                    context.cuda(), concept_mask, dbpedia_mask, concept_bitmask, dbpedia_bitmask, seed_sets, entities.cuda(
                    ), TrainType.INFER, maxlen=20, bsz=1)

            response_display, response_log = self.vector2sentence(
                preds.squeeze().detach().cpu().numpy().tolist())
            self.logs.append(response_log)
            print("Response> ", response_display)


if __name__ == '__main__':
    opt = vars(setup_args().parse_args())

    agent = IConversationalRecommender(opt)
    agent.start()
