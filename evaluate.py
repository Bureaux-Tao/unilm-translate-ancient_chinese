from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from loader import AutoTranslate, tokenizer, load_data
from model import model
from path import val_file_path
from rouge import Rouge
from utils.layers import Loss, Model
from utils.backend import keras, K


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    
    def compute_loss(self, inputs, mask = None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


output = CrossEntropy(2)(model.inputs + model.outputs)
model = Model(model.inputs, output)
model.load_weights("./weights/roformer_v2_unilm_ep26.h5")
auto_translate = AutoTranslate(start_id = None, end_id = tokenizer._token_end_id, maxlen = 128)

rouge = Rouge()
smooth = SmoothingFunction().method1


def evaluate(data, topk = 1):
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    pbar = tqdm()
    for title, content in data:
        total += 1
        title = ' '.join(title).lower()
        pred_title = ' '.join(auto_translate.generate(content, topk)).lower()
        if pred_title.strip():
            scores = rouge.get_scores(hyps = pred_title, refs = title)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references = [title.split(' ')],
                hypothesis = pred_title.split(' '),
                smoothing_function = smooth
            )
            pbar.update()
            pbar.set_description(
                'rouge_1: %.5f, rouge_2: %.5f, rouge_l: %.5f, bleu: %.5f' % (
                    rouge_1 / total, rouge_2 / total, rouge_l / total, bleu / total)
            )
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    pbar.close()
    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
    }


valid_data = load_data(val_file_path)
metrics = evaluate(valid_data)  # 评测模型
print(metrics)
