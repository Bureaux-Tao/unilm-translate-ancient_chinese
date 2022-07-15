from loader import AutoTranslate, tokenizer
from model import model
from utils.layers import Model, Loss
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
model.load_weights("./weights/unilm_roformer_ep09.h5")


def translate(test_case):
    auto_translate = AutoTranslate(start_id = None, end_id = tokenizer._token_end_id, maxlen = 128)
    pred_title = ' '.join(auto_translate.generate(test_case, 1)).lower().strip()
    print("原文:", test_case)
    print("译文:", pred_title.replace(" ", ""),'\n')


##
translate("余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。")
translate("亲贤臣，远小人，此先汉所以兴隆也；亲小人，远贤臣，此后汉所以倾颓也。")
translate("臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯。")
translate("吾日三省吾身：为人谋而不忠乎？与朋友交而不信乎？传不习乎？")
translate("其始太医以王命聚之，岁赋其二；募有能捕之者，当其租入。永之人争奔走焉。")
translate("今陛下起丰沛，收卒三千人，以之径往而卷蜀汉，定三秦，与项羽战荥阳，争成皋之口，大战七十，小战四十，使天下之民肝脑涂地，父子暴骨中野，不可胜数，哭泣之声未绝，伤痍者未起，而欲比隆於成康之时，臣窃以为不侔也。")
