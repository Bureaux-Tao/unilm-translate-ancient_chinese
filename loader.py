import json

import numpy as np

from config import maxlen
from model import model
from path import dict_path
from utils.snippets import DataGenerator, sequence_padding, AutoRegressiveDecoder
from utils.tokenizers import Tokenizer


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            l = json.loads(l)
            source, target = l['source'], l['target']
            D.append((target, source))
    return D


tokenizer = Tokenizer(dict_path, do_lower_case = True)


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                content, title, maxlen = maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class AutoTranslate(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    
    @AutoRegressiveDecoder.wraps(default_rtype = 'probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])
    
    def generate(self, text, topk = 1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen = max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk = topk)  # 基于beam search
        return tokenizer.decode(output_ids)

