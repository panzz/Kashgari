# encoding: utf-8
"""
@author: LIPan
@contact: panlee@qq.com
@version: 1.0
@time: 2019-05-24

"""
import os
import jieba
from kashgari.tasks.classification import CNNModel

FILEPATH = os.path.dirname(os.path.realpath(__file__))
FILENAME = os.path.basename(os.path.realpath(__file__))
BERT_CHN = 'D:\\ShareFolder\\_LIPan\\vobs\\_datasets\\nlp\\bert\\chinese_L-12_H-768_A-12'


def main():
    new_model = CNNModel.load_model(os.path.join(FILEPATH + '/models/thucnews_bert_model'))
    news = """「DeepMind 击败人类职业玩家的方式与他们声称的 AI 使命，以及所声称的『正确』方式完全相反。」
    DeepMind 的人工智能 AlphaStar 一战成名，击败两名人类职业选手。掌声和欢呼之余，它也引起了一些质疑。在前天 DeepMind 举办的 AMA 中，AlphaStar 项目领导者 Oriol Vinyals 和 David Silver、职业玩家 LiquidTLO 与 LiquidMaNa 回答了一些疑问。不过困惑依然存在……
    """
    x = list(jieba.cut(news))
    print(f'x:{x}')
    res = new_model.predict(x)
    print(f"res: {res}")


if __name__ == '__main__':
    # must run predict once before `app.run` to prevent predict error
    main()
