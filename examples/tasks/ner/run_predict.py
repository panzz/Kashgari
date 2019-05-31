# encoding: utf-8
"""
@author: LIPan
@contact: panlee@qq.com
@version: 1.0
@time: 2019-05-24

"""
import os
import jieba
from kashgari.tasks.seq_labeling import BLSTMCRFModel

filePath = os.path.dirname(os.path.realpath(__file__))
fileName = os.path.basename(os.path.realpath(__file__))


BERT_MODEL = os.path.join(filePath, os.path.join(
    'models', 'ber_bert_model_uncased_L12H768A12'))
print(f"filePath:{filePath}, BERT_MODEL: {BERT_MODEL}")


def main():
    # test
    new_model = BLSTMCRFModel.load_model(BERT_MODEL)
    """
    序列标注任务是中文自然语言处理（NLP）领域在句子层面中的主要任务，在给定的文本序列上预测序列中需要作出标注的标签。常见的子任务有命名实体识别（NER）、Chunk 提取以及词性标注（POS）等。
    """
    target_str = "Pretrained PyTorch models for Google's BERT, OpenAI GPT & GPT-2, Google/CMU Transformer-XL."
    print(f"target_str: {target_str}")
    # x = list(jieba.cut(target_str))
    # print(f"x:{x}")
    res = new_model.predict(target_str, output_dict=True)
    print(f"res: {res}")


if __name__ == '__main__':
    # must run predict once before `app.run` to prevent predict error
    main()
