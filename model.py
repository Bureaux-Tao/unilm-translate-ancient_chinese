from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, MODEL_TYPE, event_type
from utils.models import build_transformer_model

model = build_transformer_model(
    config_path = BASE_CONFIG_NAME,
    checkpoint_path = BASE_CKPT_NAME,
    application = event_type,
    model = MODEL_TYPE,
    return_keras_model = False
)
