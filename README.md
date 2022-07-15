# 基于Unilm的中文文言文与现代文翻译模型

## Dataset

来自[CCTC - Classical Chinese Translation Corpus](https://github.com/Scagin/CCTC)

- 《史记》（29篇）
- 《论语》（20篇全）
- 《中庸》（33章全）

原数据格式：

```json
[
    {
        "title": "",
        "contents": [
            {
                "source": "",
                "target": ""
            },
            {
                "source": "",
                "target": ""
            }
        ]
    }
]
```

|    |样本数|最大长度|最小长度|长度平均值|长度中位数|长度标准差|
|----|:---:|:------:|:-----:|:-------:|:-------:|:-------:|
|原文|7841  |180    |1      |17.67    |15.0     |13.54    |
|译文|7841  |280    |2      |29.20    |24.0     |23.28    |



需要先运行`preprocess.py`将每个`contents`中的内容转为一行一条的jsonl格式：

```json lines
...
{"source": "最骠骑将军去病，凡六出击匈奴，其四出以将军，斩捕首虏十一万馀级。", "target": "总计骠骑将军霍去病共六次出击匈奴，其中四次出击是以将军的身份，共斩获匈奴兵士十一万多人。"}
{"source": "其秋，匈奴又入雁门，杀略千馀人。", "target": "当年秋天，匈奴又攻入雁门，杀死和抢走一千余人。"}
...
```

### dataloader单条格式

- batch_token_ids: `[cls]source[sep]target[sep]`
- batch_segment_ids: `[cls]0,0,0...,0[sep]1,1,1...,1[sep]`
- batch_labels: `None`

## Project Structure

```
./
├── README.md
├── chinese_roformer-sim-char-ft_L-12_H-768_A-12        SimBert-v2模型权重
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   ├── checkpoint
│   └── vocab.txt
├── config.py                                           部分超参数
├── data
│   ├── all.jsonl                                       所有数据
│   ├── lunyu
│   │   └── part1.json
│   ├── shiji
│   │   ├── part1.json
│   │   └── part2.json
│   ├── train.jsonl                                     训练集
│   ├── valid.jsonl                                     验证集
│   └── zhongyong
│       ├── part1.json
│       └── part2.json
├── evaluate.py                                         模型评估脚本
├── loader.py                                           数据加载
├── log                                                 训练日志
│   └── nohup.out
├── model.py                                            模型搭建
├── path.py                                             所有路径
├── predict.py                                          模型预测
├── preprocess.py                                       数据预处理
├── rouge.py                                            精简过的rough包 也可以pip下载
├── statistics.py                                       文本长度统计
├── train.py                                            训练脚本
├── utils                                               bert4keras包 也可以pip下载
│   ├── __init__.py
│   ├── adversarial.py
│   ├── backend.py
│   ├── layers.py
│   ├── models.py
│   ├── optimizers.py
│   ├── snippets.py
│   └── tokenizers.py
└── weights                                             训练保存的权重
    ├── placeholder
    ├── unilm_roformer_ep09.h5
    ├── unilm_roformer_ep18.h5
    └── unilm_roformer_ep24.h5

10 directories, 48 files

```

## Model

### 上游

[SimBERT-v2](https://github.com/ZhuiyiTechnology/roformer-sim)

权重下载：

* [chinese_roformer-sim-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char_L-12_H-768_A-12.zip) 
* [chinese_roformer-sim-char_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char_L-6_H-384_A-6.zip) 
* [chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip) 
* [chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip) 

SimBERT-v2 = RoFormer + UniLM + 对比学习 + BART + 蒸馏

此预训练模型较适用于文本生成，也可做检索。

### 下游

[UniLM](https://arxiv.org/abs/1905.03197)

可以直接用单个模型做seq2seq

> UNILM直接将Seq2Seq当成句子补全来做。假如输入是“你想吃啥”，目标句子是“白切鸡”，那UNILM将这两个句子拼成一个：[CLS] 你 想 吃 啥 [SEP] 白 切 鸡 [SEP]。经过这样转化之后，最简单的方案就是训练一个语言模型，然后输入“[CLS] 你 想 吃 啥 [SEP]”来逐字预测“白 切 鸡”，直到出现“[SEP]”为止。

## Strategy

优化器：带权重衰减的Adam，学习率为2e-5

训练集验证集划分为9:1（因为bleu逐条样本评估较为费时，评估时间比训练时间长很多）

### 评估标准

1. Rouge-1: rouge-1 比较生成文本和参考文本之间的重叠词（字）数量
2. Rouge-2: rouge-2 比较生成文本和参考文本之间的 2-gram 重叠的数量
3. Rouge-L: rouge-l 根据生成文本和参考文本之间的最长公共子序列得出
4. BLEU: [Bilingual Evaluation Understudy](https://www.aclweb.org/anthology/P02-1040/)

最终以验证集bleu作为评估标准，5轮不升即停，保存验证集bleu最高的模型。Bleu评估时间较长，可先监控val_loss保存所有权重然后使用`evaluate.py`评估所有模型

## Train

```
Epoch 1/200
 - 131s - loss: 2.3976
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听了就很悲伤，孔子说：“苛政猛于虎，我曾经担心过是这样的，现在我以蒋氏的观点去看他，还是很信任他。
valid_data: {'rouge-1': 0.5842451028453086, 'rouge-2': 0.31620546417647694, 'rouge-l': 0.5568194211298347, 'bleu': 0.21229037256037345, 'best_bleu': 0.21229037256037345}

Epoch 2/200
 - 103s - loss: 2.0166
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听到这件事后越来越悲伤，孔子说：“苛政猛于虎啊！”我曾经怀疑过这件事，现在以蒋氏为观点，还是相信。
valid_data: {'rouge-1': 0.5985157071240249, 'rouge-2': 0.3366119048575613, 'rouge-l': 0.5710939585298885, 'bleu': 0.23048478239418196, 'best_bleu': 0.23048478239418196}

Epoch 3/200
 - 105s - loss: 1.7989
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听了更加悲伤，孔子说：“苛政猛于虎啊！”我曾经怀疑过这件事，现在用蒋氏来观察，还是相信的。
valid_data: {'rouge-1': 0.6002804577271128, 'rouge-2': 0.3444617290376822, 'rouge-l': 0.5738237540949773, 'bleu': 0.24237161040810848, 'best_bleu': 0.24237161040810848}

Epoch 4/200
 - 109s - loss: 1.6166
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听了更加悲哀，孔子说：“苛政真是猛于虎啊！”我曾经怀疑过这样的话，现在以蒋氏为观点，还是相信的。
valid_data: {'rouge-1': 0.6097726409999298, 'rouge-2': 0.35407884349298796, 'rouge-l': 0.5822401965756082, 'bleu': 0.24794862534222042, 'best_bleu': 0.24794862534222042}

Epoch 5/200
 - 107s - loss: 1.4571
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听了越来越悲伤，孔子说：“苛政真是猛如虎啊！”我曾经怀疑过这个问题，现在以蒋氏为观点，还是相信的。
valid_data: {'rouge-1': 0.6109039544610181, 'rouge-2': 0.3610909512745556, 'rouge-l': 0.5839454463543825, 'bleu': 0.2603792441329774, 'best_bleu': 0.2603792441329774}

Epoch 6/200
 - 107s - loss: 1.3174
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听了更加悲伤，孔子说：“苛政猛于虎啊！”我曾经怀疑过这个问题，现在以蒋氏为观点，还是相信。
valid_data: {'rouge-1': 0.6175167982622757, 'rouge-2': 0.3696226750105018, 'rouge-l': 0.5919266617963694, 'bleu': 0.2665891817415456, 'best_bleu': 0.2665891817415456}


... ...


Epoch 25/200
 - 165s - loss: 0.1325
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听说了，越是严苛的政治越猛，像虎一样！”孔子说：“我以前曾经怀疑这个问题，现在用蒋氏来看他，还是相信。
Early stop count 1/5
valid_data: {'rouge-1': 0.6297695210541802, 'rouge-2': 0.4003098830782893, 'rouge-l': 0.602590682469053, 'bleu': 0.3082482969830215, 'best_bleu': 0.3149648896381498}

Epoch 26/200
 - 168s - loss: 0.1183
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听说苛政是猛于虎的，孔子说：“苛政恐怕是猛于虎吧！”我以前曾经怀疑过这种情况，现在用蒋氏来看他，还是相信。
Early stop count 2/5
valid_data: {'rouge-1': 0.6308378294798038, 'rouge-2': 0.400053036750899, 'rouge-l': 0.6040730337830039, 'bleu': 0.3068059184122339, 'best_bleu': 0.3149648896381498}

Epoch 27/200
 - 164s - loss: 0.1082
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听说苛政是猛于虎的，孔子说：“苛政是猛于虎的！”我以前曾经怀疑过这种情况，现在用蒋氏来看他，还能相信吗？”
Early stop count 3/5
valid_data: {'rouge-1': 0.6293172902775276, 'rouge-2': 0.3994300190051216, 'rouge-l': 0.6036114878535116, 'bleu': 0.3077445379078206, 'best_bleu': 0.3149648896381498}

Epoch 28/200
 - 164s - loss: 0.0993
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听说苛政是猛虎啊！”孔子说：“苛政恐怕要比虎更猛！”我以前曾经怀疑过这种情况，现在用蒋氏来观察，还是信实。
Early stop count 4/5
valid_data: {'rouge-1': 0.6306212813424421, 'rouge-2': 0.40158040478131624, 'rouge-l': 0.6045151142206323, 'bleu': 0.310497963986462, 'best_bleu': 0.3149648896381498}

Epoch 29/200
 - 162s - loss: 0.0912
原文 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文 我听说苛政大于虎，孔子说：“苛政大于虎啊！”我曾经怀疑过这个问题，现在以蒋氏看他，还能相信。
Early stop count 5/5

Epoch 00028: early stopping THR.
valid_data: {'rouge-1': 0.6335207076843986, 'rouge-2': 0.40422218691714723, 'rouge-l': 0.6081162016522619, 'bleu': 0.31214009804785386, 'best_bleu': 0.3149648896381498}
```

## Predict

运行`predict.py`文件

```python
translate("余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。")
translate("亲贤臣，远小人，此先汉所以兴隆也；亲小人，远贤臣，此后汉所以倾颓也。")
translate("臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯。")
translate("吾日三省吾身：为人谋而不忠乎？与朋友交而不信乎？传不习乎？")
translate("其始太医以王命聚之，岁赋其二；募有能捕之者，当其租入。永之人争奔走焉。")
translate("今陛下起丰沛，收卒三千人，以之径往而卷蜀汉，定三秦，与项羽战荥阳，争成皋之口，大战七十，小战四十，使天下之民肝脑涂地，父子暴骨中野，不可胜数，哭泣之声未绝，伤痍者未起，而欲比隆於成康之时，臣窃以为不侔也。")
```

```
原文: 余闻而愈悲，孔子曰：“苛政猛于虎也！”吾尝疑乎是，今以蒋氏观之，犹信。
译文: 我听了愈发悲伤，孔子说：“苛政恐怕要比虎更猛一些！”我曾经怀疑过这个问题，现在以蒋氏为人看他，还是相信他。 

原文: 亲贤臣，远小人，此先汉所以兴隆也；亲小人，远贤臣，此后汉所以倾颓也。
译文: 亲近贤臣，疏远小人，这是先古汉朝所以兴隆的原因；亲近小人，疏远贤臣，这是后代汉朝所以倾颓的原因。 

原文: 臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯。
译文: 我穿着布衣，亲自耕作在南阳，要不牺牲性命在乱世之中度过，也不要求有什么向诸侯学习的。 

原文: 吾日三省吾身：为人谋而不忠乎？与朋友交而不信乎？传不习乎？
译文: 我每天多次地反省自己：替别人办事是不是尽心尽力呢？跟朋友交往是不是真诚，诚实呢？老师传授的知识是否复习过了呢？ 

原文: 其始太医以王命聚之，岁赋其二；募有能捕之者，当其租入。永之人争奔走焉。
译文: 这开始，太医就根据王命召集他们，每月交纳二百金，募集有能捕获他们的，就把他们租给他们，永远的人们都来奔跑。

原文: 今陛下起丰沛，收卒三千人，以之径往而卷蜀汉，定三秦，与项羽战荥阳，争成皋之口，大战七十，小战四十，使天下之民肝脑涂地，父子暴骨中野，不可胜数，哭泣之声未绝，伤痍者未起，而欲比隆於成康之时，臣窃以为不侔也。
译文: 如今陛下起事丰沛，调兵攻打，直到蜀汉，并平定了蜀汉，并与项羽交战，荥阳，争取了成皋的大口，争取了大战七十分之数，战争到成皋才发生，使天下百姓肝脑涂地，死了好几个人，哭泣的声音没有人能替他们饯行，伤残的也没有人能替他们起事，而想比成康之时，我们是不会与此相抗
```

## 训练配置

```
                 ..                    bureaux@localhost.localdomain
               .PLTJ.                  -----------------------------
              <><><><>                 OS: CentOS Linux 7 (Core) x86_64
     KKSSV' 4KKK LJ KKKL.'VSSKK        Host: 2288 V5 Purley
     KKV' 4KKKKK LJ KKKKAL 'VKK        Kernel: 3.10.0-957.el7.x86_64
     V' ' 'VKKKK LJ KKKKV' ' 'V        Uptime: 117 days, 9 hours, 14 mins
     .4MA.' 'VKK LJ KKV' '.4Mb.        Packages: 1550 (rpm)
   . KKKKKA.' 'V LJ V' '.4KKKKK .      Shell: zsh 5.0.2
 .4D KKKKKKKA.'' LJ ''.4KKKKKKK FA.    Terminal: /dev/pts/1
<QDD ++++++++++++  ++++++++++++ GFD>   CPU: Intel Xeon Silver 4214R (48) @ 2.401GHz
 'VD KKKKKKKK'.. LJ ..'KKKKKKKK FV     GPU: Intelligent Management system chip w/VGA support]
   ' VKKKKK'. .4 LJ K. .'KKKKKV '      GPU: NVIDIA 3b:00.0 NVIDIA Corporation Device 1df6
      'VK'. .4KK LJ KKA. .'KV'         Memory: 80216MiB / 128273MiB
     A. . .4KKKK LJ KKKKA. . .4
     KKA. 'KKKKK LJ KKKKK' .4KK
     KKSSA. VKKK LJ KKKV .4SSKK
              <><><><>
               'MKKM'
                 ''
```