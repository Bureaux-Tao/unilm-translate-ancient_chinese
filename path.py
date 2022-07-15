import os

event_type = "unilm"

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = current_dir

weights_path = proj_path + "/weights"
log_path = proj_path + "/log/train_log.csv"
fig_path = proj_path + "/images"

train_file_path = proj_path + "/data/train.jsonl"
val_file_path = proj_path + "/data/valid.jsonl"

# Model Config
MODEL_TYPE = 'roformer'

BASE_MODEL_DIR = proj_path + "/chinese_roformer-sim-char-ft_L-12_H-768_A-12"
BASE_CONFIG_NAME = proj_path + "/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json"
BASE_CKPT_NAME = proj_path + "/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt"
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)