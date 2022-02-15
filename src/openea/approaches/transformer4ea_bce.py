import copy
import random
import time
import os
from queue import Queue

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

import openea.modules.load.read as rd
from openea.models.trans.transe import BasicModel
from openea.modules.utils.util import load_session
from openea.modules.finding.evaluation import early_stop, test, valid
from openea.approaches.rsn4ea import BasicReader, BasicSampler
from openea.approaches.transformer import transformer_model
from openea.modules.base.losses import positive_loss
# from openea.approaches.hiea import get_relation_dict, func_r, rfunc
from openea.modules.bootstrapping.alignment_finder import search_nearest_k, mwgm, mwgm_graph_tool, check_new_alignment
from openea.modules.finding.similarity import sim
import joblib

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def func_e(e, e_dict):
    if e_dict.get(e, None):
        funce = 1 / len(e_dict[e])
    else:
        funce = 0
    return funce


def get_relation_dict(relation_triplet):
    relation_dict = {}
    for h, r, t in relation_triplet:
        r_set = relation_dict.get(r, set())
        r_set.add((h, t))
        relation_dict[r] = r_set
    return relation_dict


def func_r(r_set, arg='second'):
    r_first = [i[0] for i in r_set]
    r_first = list(set(r_first))
    r_second = [i[1] for i in r_set]
    r_second = list(set(r_second))
    if arg == 'first':
        funcr = len(r_first) / len(r_set)
    else:
        funcr = len(r_second) / len(r_set)
    return funcr


def rfunc(triple_list, ent_num, rel_num):
    head = dict()  # head of each relation
    tail = dict()  # tail of each relation
    rel_count = dict()  # count of each relation
    r_mat_ind = list()
    r_mat_val = list()
    head_r = np.zeros((ent_num, rel_num))
    tail_r = np.zeros((ent_num, rel_num))
    for triple in triple_list:
        head_r[triple[0]][triple[1]] = 1
        tail_r[triple[2]][triple[1]] = 1
        r_mat_ind.append([triple[0], triple[2]])
        r_mat_val.append(triple[1])
        if triple[1] not in rel_count:
            rel_count[triple[1]] = 1
            head[triple[1]] = set()
            tail[triple[1]] = set()
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
        else:
            rel_count[triple[1]] += 1
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
    r_mat = tf.SparseTensor(indices=r_mat_ind, values=r_mat_val, dense_shape=[ent_num, ent_num])

    return head, tail, head_r, tail_r, r_mat


def sample_neighbors(relation_triples1, relation_triples2, repeat_times, length):
    entity_neighbor_dict = dict()
    for h, r, t in list(relation_triples1 | relation_triples2):
        neighbors = entity_neighbor_dict.get(h, set())
        neighbors.add(t)
        entity_neighbor_dict[h] = neighbors

        neighbors = entity_neighbor_dict.get(t, set())
        neighbors.add(h)
        entity_neighbor_dict[t] = neighbors

    neighbor_training_data = list()
    for i in range(repeat_times):
        for ent, neighbors in entity_neighbor_dict.items():
            if len(neighbors) >= length - 1:
                samples = random.sample(neighbors, length - 1)
                neighbor_training_data.append([ent] + list(samples))
    return neighbor_training_data


def generate_label_mat(ent_num, label_list, pre_labels, soft_label_mat, soft_label, link_dict, new_link_dict=None):
    zero_val = (1 - soft_label) / (ent_num - 1)

    for i in range(len(pre_labels)):
        target = pre_labels[i]
        soft_label_mat[i, target] = zero_val
        linked_target = link_dict.get(target, 0)
        soft_label_mat[i, linked_target] = zero_val
        if new_link_dict is not None:
            if target in new_link_dict.keys():
                soft_label_mat[i, new_link_dict[target][0][0]] = zero_val

    for i in range(len(label_list)):
        target = label_list[i]
        soft_label_mat[i, target] = soft_label
        if target in link_dict.keys():
            soft_label_mat[i, link_dict[target]] = soft_label
        # todo: modify value by evidence
        if new_link_dict is not None:
            if target in new_link_dict.keys():
                # print((i, new_link_dict[target][0][0]))
                soft_label_mat[i, new_link_dict[target][0][0]] = max(new_link_dict[target][0][1][1], soft_label) / 10
    return soft_label_mat, label_list


def create_attention_mask_from_input_mask(batch_size, seq_length, to_mask):
    to_mask = to_mask[:, np.newaxis, :]
    broadcast_ones = np.ones([batch_size, seq_length, 1], dtype=np.float32)
    mask = broadcast_ones * to_mask
    return mask


class Transformer4EA(BasicReader, BasicSampler, BasicModel):
    def __init__(self):
        super().__init__()
        self._options = None
        self._train_data = None
        self._ent_paths_data = None
        self._ent_neighbors_data = None
        self._last_mean_loss = None
        self._max_size = 3
        self._best_ent_embeddings_que = Queue(maxsize=self._max_size)
        self._best_ent_embeddings = None
        self._ent_list = None
        self.ref_ent1 = None
        self.ref_ent2 = None

        self.neighbor_len = 3

    def init(self):
        self._options = opts = self.args
        opts.data_path = opts.training_data
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self.triple_list = self.kgs.kg1.relation_triples_list + self.kgs.kg2.relation_triples_list
        self.head, self.tail, self.head_r, self.tail_r, self.r_mat = rfunc(self.triple_list, self.kgs.entities_num,
                                                                           self.kgs.relations_num)

        self.read(data_path=self._options.data_path)

        ent_paths_file = '%sent_paths_%.1f_%.1f_len%s_repeat%s_%s' % (os.path.join(self._options.data_path,
                                                                                   self.args.dataset_division),
                                                                      self._options.alpha,
                                                                      self._options.beta,
                                                                      self._options.max_length,
                                                                      self._options.repeat_times,
                                                                      self._options.alignment_module)

        ent_neighbors_file = '%ssampled_ent_neighbors_len%s_repeat%s_%s' % (os.path.join(self._options.data_path,
                                                                                         self.args.dataset_division),
                                                                            self.neighbor_len,
                                                                            self._options.repeat_times,
                                                                            self._options.alignment_module)

        if not os.path.exists(ent_paths_file):
            print("sample entity paths...")
            self.sample_paths(repeat_times=self._options.repeat_times)
            ent_seq_data = self._train_data.iloc[:, [i % 2 == 0 for i in range(len(self._train_data.columns))]]
            self._ent_paths_data = ent_seq_data.drop_duplicates(keep="first")
            self._ent_paths_data.columns = list(range((self._options.max_length + 1) // 2))
            print("# entity paths:", len(self._ent_paths_data), self._ent_paths_data.columns.values)
            self._ent_paths_data.to_csv(ent_paths_file)
            print("save paths to:", ent_paths_file, self._ent_paths_data.columns, len(self._ent_paths_data))
        else:
            print('load existing entity paths from', ent_paths_file)
            self._ent_paths_data = pd.read_csv(ent_paths_file, index_col=0)

        if not os.path.exists(ent_neighbors_file):
            print("sample entity neighbors...")
            ent_neighbors = sample_neighbors(self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                                             self._options.repeat_times, self.neighbor_len)
            self._ent_neighbors_data = pd.DataFrame(ent_neighbors, dtype=np.int)
            print("# entity neighbors:", len(self._ent_neighbors_data), self._ent_neighbors_data.columns.values)
            self._ent_neighbors_data.to_csv(ent_neighbors_file)
            print("save neighbors to:", ent_neighbors_file, self._ent_neighbors_data.columns,
                  len(self._ent_neighbors_data))
        else:
            print('load existing entity neighbors from', ent_neighbors_file)
            self._ent_neighbors_data = pd.read_csv(ent_neighbors_file, index_col=0)

        self._define_variables()
        self._define_embed_graph()
        self._define_triple_graph()
        self._define_neighbor_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def _define_variables(self, initializer_range=0.02):
        self._ent_list = list(range(self._ent_num))

        # The embedding self.ent_embeds[self._ent_num, ] is for the mask.
        self.ent_embeds = tf.get_variable('entity_embedding', [self._ent_num + 1, self._options.hidden_size],
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.rel_embeds = tf.get_variable('relation_embedding', [self.kgs.relations_num, self._options.hidden_size],
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self._lr = tf.Variable(self._options.learning_rate, trainable=False)
        self._optimizer = tf.train.AdamOptimizer(self._options.learning_rate)
        self._triple_optimizer = tf.train.AdamOptimizer(self._options.learning_rate)

    def transformer_encoder(self, em_seq):
        em_seq = tf.nn.dropout(em_seq, keep_prob=1 - self._options.input_dropout_prob)
        with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
            outputs = transformer_model(input_tensor=em_seq,
                                        attention_mask=None,
                                        hidden_size=self.args.hidden_size,
                                        num_hidden_layers=self.args.num_layers,
                                        num_attention_heads=self.args.num_attention_heads,
                                        intermediate_size=self.args.hidden_size * 4,
                                        hidden_dropout_prob=self._options.former_dropout_prob,
                                        attention_probs_dropout_prob=self._options.former_dropout_prob,
                                        do_return_all_layers=False)
        return outputs

    def bce_loss(self, inputs, soft_targets):
        ent_embeds = self.ent_embeds[0:self._ent_num, :]
        # ent_embeds = tf.nn.embedding_lookup(self.ent_embeds, self._ent_list)
        logs = tf.matmul(inputs, ent_embeds, transpose_b=True)
        log_softmax = tf.nn.log_softmax(logs + 1e-8)
        return tf.reduce_mean(tf.reduce_sum(- soft_targets * log_softmax, axis=1))

    def build_path_graph(self, length=15, reuse=False):
        options = self._options
        batch_size = options.batch_size
        ent_seq_len = (length + 1) // 2

        seq = tf.placeholder(tf.int32, [batch_size, ent_seq_len], name='seq' + str(length))
        label = tf.placeholder(tf.float32, [batch_size, self._ent_num], name='lab' + str(length))
        attention_mask = tf.placeholder(tf.float32, [batch_size, ent_seq_len, ent_seq_len], name='attention_mask')
        mask_position = tf.placeholder(tf.int32, name='mask_position')

        ent_em = tf.nn.embedding_lookup(self.ent_embeds, seq)

        with tf.variable_scope('transformer_encoder', reuse=reuse):
            outputs = self.transformer_encoder(ent_em)

        mask_em = outputs[:, mask_position, :]
        seq_loss = self.bce_loss(mask_em, label)

        return seq_loss, seq, label, mask_position, attention_mask

    def build_neighbor_graph(self, reuse=False):
        options = self._options
        batch_size = options.batch_size
        ent_seq_len = self.neighbor_len

        seq = tf.placeholder(tf.int32, [batch_size, ent_seq_len], name='neighbor' + str(ent_seq_len))
        label = tf.placeholder(tf.float32, [batch_size, self._ent_num], name='neighbor_lab' + str(ent_seq_len))
        attention_mask = tf.placeholder(tf.float32, [batch_size, ent_seq_len, ent_seq_len], name='neighbor_mask')
        mask_position = tf.placeholder(tf.int32, name='neighbor_mask_position')

        ent_em = tf.nn.embedding_lookup(self.ent_embeds, seq)

        with tf.variable_scope('neighbor_transformer_encoder', reuse=reuse):
            outputs = self.transformer_encoder(ent_em)

        mask_em = outputs[:, mask_position, :]
        seq_loss = self.bce_loss(mask_em, label)

        return seq_loss, seq, label, mask_position, attention_mask

    def build_alignment_graph(self, neg_margin):
        seed1 = tf.placeholder(tf.int32, shape=[None])
        seed2 = tf.placeholder(tf.int32, shape=[None])
        seed1_embeds = tf.nn.embedding_lookup(self.ent_embeds, seed1)
        seed2_embeds = tf.nn.embedding_lookup(self.ent_embeds, seed2)
        seed1_embeds = tf.nn.l2_normalize(seed1_embeds, 1)
        seed2_embeds = tf.nn.l2_normalize(seed2_embeds, 1)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(seed1_embeds - seed2_embeds), axis=1))

        neg_seed1 = tf.placeholder(tf.int32, shape=[None])
        neg_seed2 = tf.placeholder(tf.int32, shape=[None])
        neg_seed1_embeds = tf.nn.embedding_lookup(self.ent_embeds, neg_seed1)
        neg_seed2_embeds = tf.nn.embedding_lookup(self.ent_embeds, neg_seed2)
        neg_seed1_embeds = tf.nn.l2_normalize(neg_seed1_embeds, 1)
        neg_seed2_embeds = tf.nn.l2_normalize(neg_seed2_embeds, 1)

        neg_loss3 = tf.reduce_mean(tf.nn.relu(tf.constant(neg_margin) - tf.reduce_sum(
            tf.square(neg_seed1_embeds - neg_seed2_embeds), axis=1)))
        neg_loss = neg_loss3

        return loss + neg_loss, seed1, seed2, neg_seed1, neg_seed2

    def _define_embed_graph(self):
        self._path_loss, self._seq, self._label, self._mask_position, self._attention_mask = \
            self.build_path_graph(length=self._options.max_length, reuse=False)

        self._align_loss, self._seed1, self._seed2, self._neg_seed1, self._neg_seed2 = \
            self.build_alignment_graph(self._options.neg_margin)

        self._loss = self._path_loss + self._align_loss

        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, train_vars), 2.0)
        # grads = tf.gradients(self._loss, train_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._train_op = self._optimizer.apply_gradients(zip(grads, train_vars),
                                                             global_step=tf.train.get_or_create_global_step())

    def _define_neighbor_graph(self):
        self._neighbor_loss, self._neighbor, self._neighbor_label, self._neighbor_mask_position, \
        self._neighbor_attention_mask = self.build_neighbor_graph(reuse=True)

        # self._align_loss, self._seed1, self._seed2, self._neg_seed1, self._neg_seed2 = \
        #     self.build_alignment_graph(self._options.neg_margin)

        self._neighbor_loss = self._neighbor_loss + self._align_loss

        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._neighbor_loss, train_vars), 2.0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._neighbor_train_op = self._optimizer.apply_gradients(zip(grads, train_vars),
                                                                       global_step=tf.train.get_or_create_global_step())

    def _define_triple_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            phs = tf.nn.l2_normalize(phs, axis=1)
            prs = tf.nn.l2_normalize(prs, axis=1)
            pts = tf.nn.l2_normalize(pts, axis=1)
        with tf.name_scope('triple_loss'):
            self._triple_loss = positive_loss(phs, prs, pts, "L2")

            train_vars = tf.trainable_variables()
            grads = tf.gradients(self._triple_loss, train_vars)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._triple_train_op = self._triple_optimizer. \
                    apply_gradients(zip(grads, train_vars), global_step=tf.train.get_or_create_global_step())

    def _get_ent_embeds(self):
        if self._best_ent_embeddings_que.qsize() == 0:
            return preprocessing.normalize(self._best_ent_embeddings)
        total_best_embeds = preprocessing.normalize(self._best_ent_embeddings)
        n = 1
        while self._best_ent_embeddings_que.qsize() > 0:
            n += 1
            total_best_embeds += preprocessing.normalize(self._best_ent_embeddings_que.get())
        print("lookup best entity embeddings of", n)
        return total_best_embeds / n

    def _eval_test_embeddings(self):
        best_ent_embeddings = self._get_ent_embeds()
        embeds1 = best_ent_embeddings[self.kgs.test_entities1, :]
        embeds2 = best_ent_embeddings[self.kgs.test_entities2, :]
        return embeds1, embeds2, None

    def eval_ref_sim_mat(self):
        refs1_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent1)
        refs2_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent2)
        refs1_embeddings = tf.nn.l2_normalize(refs1_embeddings, 1)
        refs2_embeddings = tf.nn.l2_normalize(refs2_embeddings, 1)
        mat_op = tf.matmul(refs1_embeddings, refs2_embeddings, transpose_b=True)
        sim_mat = self.session.run(mat_op)
        return sim_mat

    def test(self, save=False):
        embeds1, embeds2, mapping = self._eval_test_embeddings()
        rest_12, _, _ = test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)

    def valid(self, stop_metric):
        embeds1, embeds2, mapping = self._eval_valid_embeddings()
        hits1_12, mrr_12 = valid(embeds1, embeds2, mapping, self.args.top_k,
                                 self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def save(self):
        if self._options.is_save:
            best_ent_embeddings = self._get_ent_embeds()
            rd.save_embeddings(self.out_folder, self.kgs, best_ent_embeddings, None, None, mapping_mat=None)

    def save_ent_embeds(self):
        if self._best_ent_embeddings_que.qsize() >= self._max_size:
            self._best_ent_embeddings_que.get()
        self._best_ent_embeddings_que.put(self.ent_embeds.eval(session=self.session))
        assert self._best_ent_embeddings_que.qsize() <= self._max_size

    def seq_train(self, data, win_range, link_dict, new_link_dict):
        num_batch = len(data) // self._options.batch_size
        ent_seq_len = (self._options.max_length + 1) // 2
        link_batch_size = max(ent_seq_len * len(self.kgs.train_links) // num_batch, 1)

        neg_link_batch_size = self._options.neg_samples * link_batch_size
        neg_link_batch_size = min(neg_link_batch_size, len(self.kgs.test_entities1 + self.kgs.valid_entities1))

        zero_label = (1 - self._options.soft_label) / (self._ent_num - 1)
        label_mat = np.full((self._options.batch_size, self._ent_num), zero_label, dtype=np.float)

        fetches = {'loss': self._loss, 'train_op': self._train_op}
        loss = 0
        batch_time = 0.0

        pre_batch_labels = []

        for mask_position in win_range:
            choices = np.random.choice(len(data), size=len(data), replace=True)
            for i in range(num_batch):
                time_start = time.time()

                one_batch_choices = choices[i * self._options.batch_size: (i + 1) * self._options.batch_size]
                one_batch_data = data.iloc[one_batch_choices]
                seq = one_batch_data.values[:, :ent_seq_len]

                labels = seq[:, mask_position].tolist()

                # get_label_mat time = 0.00409 s.
                label_mat, pre_batch_labels = generate_label_mat(self._ent_num, labels, pre_batch_labels, label_mat,
                                                                 self._options.soft_label, link_dict, new_link_dict)

                # mask_seq time = 0.00029 s.
                mask_seq = copy.deepcopy(seq)
                mask_seq[::, mask_position] = [self._ent_num] * self._options.batch_size

                link_batch = random.sample(self.kgs.train_links, link_batch_size)
                seed1 = [link[0] for link in link_batch]
                seed2 = [link[1] for link in link_batch]

                neg_seed1 = random.sample(self.kgs.valid_entities1 + self.kgs.test_entities1, neg_link_batch_size)
                neg_seed2 = random.sample(self.kgs.valid_entities2 + self.kgs.test_entities2, neg_link_batch_size)

                # session.run time = 0.89731 s.
                feed_dict = {self._seq: mask_seq, self._label: label_mat, self._mask_position: mask_position,
                             self._seed1: seed1, self._seed2: seed2,
                             self._neg_seed1: neg_seed1, self._neg_seed2: neg_seed2}
                rest = self.session.run(fetches, feed_dict)

                batch_time = time.time() - time_start
                loss += rest['loss']
                self._last_mean_loss = loss

        return self._last_mean_loss, batch_time

    def neighbor_train(self, data, win_range, link_dict, new_link_dict):
        num_batch = len(data) // self._options.batch_size
        ent_seq_len = self.neighbor_len
        link_batch_size = max(ent_seq_len * len(self.kgs.train_links) // num_batch, 1)

        neg_link_batch_size = self._options.neg_samples * link_batch_size
        neg_link_batch_size = min(neg_link_batch_size, len(self.kgs.test_entities1 + self.kgs.valid_entities1))

        zero_label = (1 - self._options.soft_label) / (self._ent_num - 1)
        label_mat = np.full((self._options.batch_size, self._ent_num), zero_label, dtype=np.float)

        fetches = {'loss': self._neighbor_loss, 'train_op': self._neighbor_train_op}
        loss = 0
        batch_time = 0.0

        pre_batch_labels = []

        for mask_position in win_range:
            choices = np.random.choice(len(data), size=len(data), replace=True)
            for i in range(num_batch):
                time_start = time.time()

                one_batch_choices = choices[i * self._options.batch_size: (i + 1) * self._options.batch_size]
                one_batch_data = data.iloc[one_batch_choices]
                seq = one_batch_data.values[:, :ent_seq_len]

                labels = seq[:, mask_position].tolist()

                # get_label_mat time = 0.00409 s.
                label_mat, pre_batch_labels = generate_label_mat(self._ent_num, labels, pre_batch_labels, label_mat,
                                                                 self._options.soft_label, link_dict, new_link_dict)

                # mask_seq time = 0.00029 s.
                mask_seq = copy.deepcopy(seq)
                mask_seq[::, mask_position] = [self._ent_num] * self._options.batch_size

                link_batch = random.sample(self.kgs.train_links, link_batch_size)
                seed1 = [link[0] for link in link_batch]
                seed2 = [link[1] for link in link_batch]

                neg_seed1 = random.sample(self.kgs.valid_entities1 + self.kgs.test_entities1, neg_link_batch_size)
                neg_seed2 = random.sample(self.kgs.valid_entities2 + self.kgs.test_entities2, neg_link_batch_size)

                # session.run time = 0.89731 s.
                feed_dict = {self._neighbor: mask_seq, self._neighbor_label: label_mat,
                             self._neighbor_mask_position: mask_position,
                             self._seed1: seed1, self._seed2: seed2,
                             self._neg_seed1: neg_seed1, self._neg_seed2: neg_seed2}
                rest = self.session.run(fetches, feed_dict)

                batch_time = time.time() - time_start
                loss += rest['loss']
                self._last_mean_loss = loss

        return self._last_mean_loss, batch_time

    def triple_train(self, triples):
        start = time.time()
        epoch_loss = 0
        num_batch = len(triples) // self._options.triple_batch_size
        for i in range(num_batch):
            batch_pos = random.sample(triples, self._options.batch_size)
            batch_loss, _ = self.session.run(fetches=[self._triple_loss, self._triple_train_op],
                                             feed_dict={self.pos_hs: [x[0] for x in batch_pos],
                                                        self.pos_rs: [x[1] for x in batch_pos],
                                                        self.pos_ts: [x[2] for x in batch_pos]})
            epoch_loss += batch_loss
        epoch_loss /= len(triples)
        return epoch_loss, time.time() - start

    def head_tail(self, h_i, h_j, t_i, t_j, hi_num, hj_num, ti_num, tj_num):
        h_sim = sim(h_i, h_j, metric='cosine')
        thre_h = min(self.args.thre_h, (np.max(h_sim) + np.mean(h_sim)) / 2)
        t_sim = sim(t_i, t_j, metric='cosine')
        thre_t = min(self.args.thre_t, (np.max(t_sim) + np.mean(t_sim)) / 2)
        h_sim = h_sim.max(axis=0 if h_sim.shape[0] > h_sim.shape[1] else 1)
        h_sim = h_sim * (h_sim > thre_h)  # todo: filter by threshold or not
        t_sim = t_sim.max(axis=0 if t_sim.shape[0] > t_sim.shape[1] else 1)
        t_sim = t_sim * (t_sim > thre_t)
        numerator = sum(h_sim) + sum(t_sim)
        denominator = hi_num + hj_num - sum(h_sim) + ti_num + tj_num - sum(t_sim)
        return numerator / denominator

    def head_tail_cat(self, h_i, h_j, t_i, t_j, hi_num, hj_num, ti_num, tj_num):
        h_sim = sim(h_i, h_j, metric='cosine')
        t_sim = sim(t_i, t_j, metric='cosine')
        return (h_sim + t_sim) / 2

    def calculate_holistic_mat(self, knn, sim_mat):
        rel_embeds = self.session.run(self.rel_embeds)
        ent_embeds = self.session.run(self.ent_embeds)
        rel_prob = self.rel_prob(rel_embeds)
        knn = list(knn)
        prob_e_mat = []
        for i in range(len(knn)):
            a = self.ref_ent1[knn[i][0]]
            b = self.ref_ent2[knn[i][1]]
            prob_e = self.pr_e([a, b], self.kgs, ent_embeds, rel_prob, self.r_func1, self.r_func2,
                               self.kgs.kg1.rel_index, self.kgs.kg2.rel_index, self.kgs.relations_num)
            prob_e_mat.append([sim_mat[knn[i][0], knn[i][1]], prob_e])
        return prob_e_mat

    def rel_prob(self, rel_embeds):
        test_r = [self.kgs.kg1.relations_list, self.kgs.kg2.relations_list]
        print("prob_r calculating...")
        t1 = time.time()
        inlayer = self.session.run(self.ent_embeds)
        prob_r, rsim_dict = self.pr_r(test_r, self.head, self.tail, inlayer) 
        rel_prob = prob_r
        print('prob_r cost： {:.4f}s'.format(time.time() - t1))
        return rel_prob

    def pr_r(self, test_pair, head, tail, inlayer, rel_num=346):
        r2e = {}
        for ill in test_pair[0] + test_pair[1]:
            r2e[ill] = [head.get(ill, set()), tail.get(ill, set())]

        rpairs = {}
        r_dict = {}
        for i in test_pair[0]:
            for j in test_pair[1]:
                # pair-wise sim of connected entities of relations.
                h_i_e = np.array([inlayer[e] for e in r2e[i][0]])
                efunc_hi = np.array([func_e(e, self.kgs.kg1.rt_dict) for e in r2e[i][0]]).reshape(-1, 1)
                efunc_hi = efunc_hi / sum(efunc_hi)
                h_i = np.sum(np.multiply(efunc_hi, h_i_e), 0).reshape(1, -1)
                t_i_e = np.array([inlayer[e] for e in r2e[i][1]])
                efunc_ti = np.array([func_e(e, self.kgs.kg1.hr_dict) for e in r2e[i][1]]).reshape(-1, 1)
                efunc_ti = efunc_ti / sum(efunc_ti)
                t_i = np.sum(np.multiply(efunc_ti, t_i_e), 0).reshape(1, -1)
                h_j_e = np.array([inlayer[e] for e in r2e[j][0]])
                efunc_hj = np.array([func_e(e, self.kgs.kg2.rt_dict) for e in r2e[j][0]]).reshape(-1, 1)
                efunc_hj = efunc_hj / sum(efunc_hj)
                h_j = np.sum(np.multiply(efunc_hj, h_j_e), 0).reshape(1, -1)
                t_j_e = np.array([inlayer[e] for e in r2e[j][1]])
                efunc_tj = np.array([func_e(e, self.kgs.kg2.hr_dict) for e in r2e[j][1]]).reshape(-1, 1)
                efunc_tj = efunc_tj / sum(efunc_tj)
                t_j = np.sum(np.multiply(efunc_tj, t_j_e), 0).reshape(1, -1)
                a1 = self.head_tail_cat(h_i, h_j, t_i, t_j, len(r2e[i][0]), len(r2e[j][0]), len(r2e[i][1]),
                                        len(r2e[j][1]))
                a2 = self.head_tail_cat(t_i, h_j, h_i, t_j, len(r2e[i][1]), len(r2e[j][0]), len(r2e[i][0]),
                                        len(r2e[j][1]))
                a3 = self.head_tail_cat(h_i, t_j, t_i, h_j, len(r2e[i][0]), len(r2e[j][1]), len(r2e[i][1]),
                                        len(r2e[j][0]))
                a4 = self.head_tail_cat(t_i, t_j, h_i, h_j, len(r2e[i][1]), len(r2e[j][0]), len(r2e[i][0]),
                                        len(r2e[j][1]))
                rpairs[(i, j)] = [a1, a2, a3, a4]
                r_dict[(i, j)], r_dict[i + rel_num, j], r_dict[i, j + rel_num], r_dict[
                    i + rel_num, j + rel_num] = a1, a2, a3, a4

        coinc1 = []
        for row in test_pair[0]:
            list = []
            for col in test_pair[1]:
                list.append(rpairs[(row, col)][0])
            for col in test_pair[1]:
                list.append(rpairs[(row, col)][2])
            coinc1.append(list)
        coinc2 = []
        for row in test_pair[0]:
            list = []
            for col in test_pair[1]:
                list.append(rpairs[(row, col)][1])
            for col in test_pair[1]:
                list.append(rpairs[(row, col)][3])
            coinc2.append(list)

        coinc = np.concatenate((np.array(coinc1), np.array(coinc2)), axis=0)
        return coinc, r_dict

    def pr_e(self, test_pair, kgs, e_emb, rel_sim, r_func1, r_func2, rel_index1, rel_index2, rel_num=0):
        x, y = int(test_pair[0]), int(test_pair[1])
        rt1 = kgs.kg1.rt_dict.get(x, set())
        hr1 = kgs.kg1.hr_dict.get(x, set())
        r1_out = [r[0] for r in rt1]
        r1_in = [r[1] + rel_num for r in hr1]
        r1_out_fun = np.array([r_func1[i[0]][1] for i in rt1])
        r1_in_fun = np.array([r_func1[i[1]][0] for i in hr1])
        e1_out = np.array([e_emb[e[1]] for e in rt1])
        e1_in = np.array([e_emb[e[0]] for e in hr1])
        if len(rt1) == 0 and len(hr1) == 0:
            return 0
        if len(rt1) == 0:
            r1 = r1_in
            r1_fun = r1_in_fun
            e1 = e1_in
        elif len(hr1) == 0:
            r1 = r1_out
            r1_fun = r1_out_fun
            e1 = e1_out
        else:
            r1 = r1_out + r1_in
            r1_fun = np.concatenate((r1_out_fun, r1_in_fun))
            e1 = np.concatenate((e1_out, e1_in))

        rt2 = kgs.kg2.rt_dict.get(y, set())
        hr2 = kgs.kg2.hr_dict.get(y, set())
        r2_out = [r[0] for r in rt2]
        r2_in = [r[1] + rel_num for r in hr2]
        r2_out_fun = np.array([r_func2[i[0]][1] for i in rt2])
        r2_in_fun = np.array([r_func2[i[1]][0] for i in hr2])
        e2_out = np.array([e_emb[e[1]] for e in rt2])
        e2_in = np.array([e_emb[e[0]] for e in hr2])
        if len(rt2) == 0 and len(hr2) == 0:
            return 0
        if len(rt2) == 0:
            r2 = r2_in
            r2_fun = r2_in_fun
            e2 = e2_in
        elif len(hr2) == 0:
            r2 = r2_out
            r2_fun = r2_out_fun
            e2 = e2_out
        else:
            r2 = r2_out + r2_in
            r2_fun = np.concatenate((r2_out_fun, r2_in_fun))
            e2 = np.concatenate((e2_out, e2_in))

        r_match = np.zeros((len(r1), len(r2)))
        for a in range(len(r1)):
            for b in range(len(r2)):
                aa, bb = rel_index1.get(r1[a]), rel_index2.get(r2[b])
                r_match[a, b] = rel_sim[aa, bb]
        rfun_match = np.matmul(r1_fun.reshape(-1, 1), r2_fun.reshape(1, -1))
        # pick max val of each relation
        e_match = sim(e1, e2, metric='cosine')
        re_match = np.multiply(r_match, e_match)
        mask = (re_match.max(axis=0 if re_match.shape[0] > re_match.shape[1] else 1, keepdims=1) == re_match)
        re_match = mask * re_match
        thre_re = max(self.args.thre_re, (np.max(re_match) + np.mean(re_match)) / 2)
        re_match = re_match * (re_match > thre_re)  # todo: threshold need adjust. maybe need to learn
        re_match = np.multiply(re_match, rfun_match)  # combine relation importance
        numerator = sum(sum(re_match))
        denominator = sum(sum(np.multiply(rfun_match, mask)))
        return numerator / denominator

    def run(self):
        t = time.time()
        best_rest = 0.0

        link_dict = dict()
        for e1, e2 in self.kgs.train_links:
            link_dict[e1] = e2
            link_dict[e2] = e1

        # generate relation func
        relation_dict1 = get_relation_dict(self.kgs.kg1.relation_triples_list)
        relation_dict2 = get_relation_dict(self.kgs.kg2.relation_triples_list)
        self.r_func1 = {}
        for key in relation_dict1:
            fun = func_r(relation_dict1.get(key), arg='first')
            ifun = func_r(relation_dict1.get(key), arg='second')
            self.r_func1[key] = [fun, ifun]
        self.r_func2 = {}
        for key in relation_dict2:
            fun = func_r(relation_dict2.get(key), arg='first')
            ifun = func_r(relation_dict2.get(key), arg='second')
            self.r_func2[key] = [fun, ifun]
        self.kgs.kg1.rel_index, self.kgs.kg2.rel_index = {}, {}
        for i in range(len(self.kgs.kg1.relations_list)):
            self.kgs.kg1.rel_index[self.kgs.kg1.relations_list[i]] = i
            self.kgs.kg1.rel_index[self.kgs.kg1.relations_list[i] + self.kgs.relations_num] = \
                i + len(self.kgs.kg1.relations_list)
        for i in range(len(self.kgs.kg2.relations_list)):
            self.kgs.kg2.rel_index[self.kgs.kg2.relations_list[i]] = i
            self.kgs.kg2.rel_index[self.kgs.kg2.relations_list[i] + self.kgs.relations_num] = \
                i + len(self.kgs.kg2.relations_list)
        ground_pair = self.kgs.train_links + self.kgs.valid_links + self.kgs.test_links

        ent_seq_len = (self._options.max_length + 1) // 2
        win_start = self._options.win_start
        assert 2 * win_start < ent_seq_len
        win_range = list(range(win_start, ent_seq_len - win_start))

        for i in range(1, self.args.max_epoch + 1):

            if i > self.args.start_inference and i % self.args.eval_freq == 0:
                sim_mat = self.eval_ref_sim_mat()
                nearest_k_neighbors = search_nearest_k(sim_mat, self.args.k)
                prob_e_mat = self.calculate_holistic_mat(nearest_k_neighbors, sim_mat)
                joblib.dump(nearest_k_neighbors, os.path.join(self._options.data_path, self.args.dataset_division) +
                            'nearest_k_neighbors_' + str(i))
                joblib.dump(prob_e_mat, os.path.join(self._options.data_path, self.args.dataset_division) +
                            'prob_e_mat_' + str(i))

                pos_set = []
                pos_score = []
                neg_set = []
                neg_score = []
                for index in range(len(nearest_k_neighbors)):
                    if prob_e_mat[index][1] > self.args.pos_thre:
                        pos_set.append(list(nearest_k_neighbors)[index])
                        pos_score.append(prob_e_mat[index])
                    if prob_e_mat[index][1] < self.args.neg_thre:
                        neg_set.append(list(nearest_k_neighbors)[index])
                        neg_score.append(prob_e_mat[index])
                if len(pos_set) != 0:
                    pos_set = mwgm(pos_set, sim_mat, mwgm_graph_tool)
                check_new_alignment(pos_set, context="after mwgm")
                new_link_dict = dict()
                new_link_paris = set()
                for ii, pair in enumerate(pos_set):
                    new_link_paris.add((self.ref_ent1[pair[0]], self.ref_ent2[pair[1]]))
                    if new_link_dict.get(self.ref_ent1[pair[0]], None) is None:
                        new_link_dict[self.ref_ent1[pair[0]]] = [(self.ref_ent2[pair[1]], pos_score[ii])]
                    else:
                        new_link_dict[self.ref_ent1[pair[0]]].append((self.ref_ent2[pair[1]], pos_score[ii]))
                dic_path = os.path.join(self._options.data_path,
                                        self.args.dataset_division) + 'trans4ea_new_link_dict_' + str(i)
                print(dic_path)
                print("new_link_paris", len(new_link_paris), len(new_link_paris & set(ground_pair)))
                joblib.dump(new_link_dict, dic_path)
            else:
                new_link_dict = None

            if new_link_dict is not None and len(new_link_dict) > 0:
                start = time.time()
                col = random.choice(win_range)
                loss, t = self.seq_train(self._ent_paths_data, [col], link_dict, new_link_dict)
                print('epoch %i, loss: %.2f, batch time: %.2fs, total time: %.1fs' % (i, loss, t, time.time() - start))

            time_start = time.time()
            loss1, time1 = self.seq_train(self._ent_paths_data, win_range, link_dict, None)
            # loss2, time2 = self.neighbor_train(self._ent_neighbors_data, [0], link_dict, None)
            # loss3, time3 = self.triple_train(self.kgs.kg1.relation_triples_list +
            #                                  self.kgs.kg2.relation_triples_list)
            loss2, time2 = 0., 0.
            loss3, time3 = 0., 0.
            print('epoch %i, loss: %.2f, batch time: %.2fs, total time: %.1fs' % (i, loss1 + loss2 + loss3,
                                                                                  time1 + time2 + time3,
                                                                                  time.time() - time_start))

            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i >= self.args.max_epoch:
                    break
                self.save_ent_embeds()
                if flag > best_rest:
                    self._best_ent_embeddings = self.ent_embeds.eval(session=self.session)
                    best_rest = flag

        print("Training ends. Total time = {:.1f} s.".format(time.time() - t))
