from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from config import batch_size, epochs
from loader import load_data, AutoTranslate, tokenizer, data_generator
from model import model
from path import train_file_path, val_file_path, weights_path, event_type, MODEL_TYPE
from utils.backend import keras, K
from utils.layers import Loss
from utils.optimizers import Adam, extend_with_exponential_moving_average, extend_with_weight_decay
from keras.models import Model
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

train_data = load_data(train_file_path)
valid_data = load_data(val_file_path)


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
# AdamEMA = extend_with_exponential_moving_average(Adam, name = 'AdamEMA')
# optimizer = AdamEMA(lr = 2e-5)
# optimizer = Adam(2e-5)
AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(lr = 2e-5, weight_decay_rate = 0.01)

model.compile(optimizer = optimizer)
model.summary()

auto_translate = AutoTranslate(start_id = None, end_id = tokenizer._token_end_id, maxlen = 128)
count_model_did_not_improve = 0


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self, patience = 5, silent = False):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.
        self.patience = patience
        self.silent = silent
    
    def on_epoch_end(self, epoch, logs = None):
        global count_model_did_not_improve
        # 保存最优
        save_file_path = "{}/{}_{}_best.h5".format(weights_path, event_type, MODEL_TYPE)
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            logs['best_bleu'] = self.best_bleu
            model.save_weights(save_file_path)  # 保存模型
            count_model_did_not_improve = 0
        else:
            count_model_did_not_improve += 1
            print("Early stop count " + str(count_model_did_not_improve) + "/" + str(self.patience))
            if count_model_did_not_improve >= self.patience:
                self.model.stop_training = True
                print("Epoch %05d: early stopping THR." % epoch)
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)
    
    def evaluate(self, data, topk = 1):
        # optimizer.apply_ema_weights()
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        if not self.silent:
            pbar = tqdm()
        for title, content in data:
            total += 1
            title = ' '.join(title).lower()
            pred_title = ' '.join(auto_translate.generate(content, topk)).lower()
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps = pred_title, refs = title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references = [title.split(' ')],
                    hypothesis = pred_title.split(' '),
                    smoothing_function = self.smooth
                )
                if not self.silent:
                    pbar.update()
                    pbar.set_description(
                        'rouge_1: %.5f, rouge_2: %.5f, rouge_l: %.5f, bleu: %.5f' % (
                            rouge_1 / total, rouge_2 / total, rouge_l / total, bleu / total)
                    )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        # optimizer.reset_old_weights()
        if not self.silent:
            pbar.close()
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


class TestCaseOutput(keras.callbacks.Callback):
    def __init__(self, test_case):
        self.test_case = test_case
    
    def on_epoch_end(self, epoch, logs = None):
        pred_title = ' '.join(auto_translate.generate(self.test_case, 1)).lower().strip()
        print("原文", self.test_case)
        print("译文", pred_title.replace(" ", ""))


if __name__ == '__main__':
    evaluator = Evaluator(patience = 5, silent = True)
    test_case_output = TestCaseOutput("余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。")
    save_file_path = ("{}/{}_{}".format(weights_path, event_type, MODEL_TYPE)) + "_ep{epoch:02d}.h5"
    # early_stopping = EarlyStopping(monitor = 'best_bleu', patience = 5, verbose = 1)  # 提前结束
    save_model = ModelCheckpoint(save_file_path, monitor = 'loss', verbose = 0, period = 1,
                                 save_weights_only = True, save_best_only = False)
    train_generator = data_generator(train_data, batch_size)
    # val_generator = data_generator(valid_data, batch_size)
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        # validation_data = val_generator.forfit(),
        # validation_steps = len(val_generator),
        epochs = epochs,
        callbacks = [test_case_output, evaluator, save_model],
        # callbacks = [early_stopping, test_case_output, save_model],
        verbose = 2
    )
