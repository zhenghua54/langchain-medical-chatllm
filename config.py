import os
from mindspore import context
from mindspore import nn

# 设置运行设备和模式
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU" if context.get_context("device_target") == "GPU" else "CPU")
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

# 设备配置
EMBEDDING_DEVICE = context.get_context("device_target")
LLM_DEVICE = context.get_context("device_target")

MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')

VS_ROOT_PATH = './'
# MindSpore 目前不直接支持查询GPU数量的API，需要用户根据实际情况设置
num_gpus = 1  # 示例值，根据实际情况调整
init_base = "Graph"
# Graph database
host = '10.249.7.4'  # http://10.90.1.19:22075/
port = '7687'
user = 'neo4j'
pwd = 'password'
# 初始化模型配置
init_llm = "BaiChuan2-13B-Chat-4bits"
init_embedding_model = "text2vec-base"

# 模型配置
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "text2vec-base": "/datasets/text2vec-base-chinese",  # 有
    'simbert-base-chinese': 'WangZeJun/simbert-base-chinese'
}
quant8_saved_dir = "baichuan-inc/Baichuan2-13B-Chat"

llm_model_dict = {
    "chatglm": {
        "ChatGLM-6B": "THUDM/chatglm-6b",
        "ChatGLM-6B-int4": "THUDM/chatglm-6b-int4",
        "ChatGLM-6B-int8": "/datasets/chatglm-6b-int8",  # 有
        "ChatGLM-6b-int4-qe": "THUDM/chatglm-6b-int4-qe"
    },
    "belle": {
        "BELLE-LLaMA-Local": "/pretrainmodel/belle",
    },
    "vicuna": {
        "Vicuna-Local": "/pretrainmodel/vicuna",
    },
    "baichuan": {
        "BaiChuan2-7B": "baichuan-inc/Baichuan2-7B-Chat",
        "BaiChuan2-13B-Chat": "./path/to/my_local_model/Baichuan2-13B-Chat",
        "BaiChuan2-13B-Chat-4bits": "./path/to/my_local_model/Baichuan2-13B-Chat-4bits",
        "BaiChuan2-13B-Chat-Int4": "/datasets/baichuan2-13B-chat-int4"  # 有
    }
}