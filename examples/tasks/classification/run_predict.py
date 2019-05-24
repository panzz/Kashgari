# encoding: utf-8
"""
@author: LIPan
@contact: panlee@qq.com
@version: 1.0
@time: 2019-05-24

"""
import random
import keras

from kashgari.tasks.classification import CNNLSTMModel
from kashgari.corpus import ChinaPeoplesDailyNerCorpus
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel
from kashgari.utils.logger import init_logger
init_logger()

BERT_CHN = 'D:\\ShareFolder\\_LIPan\\vobs\\_datasets\\nlp\\bert\\chinese_L-12_H-768_A-12'


def main():
    train_x, train_y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(
        'train')
    val_x, val_y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(
        'validate')
    test_x, test_y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(
        'test')

    print(f"train data count: {len(train_x)}, {len(train_y)}")
    print(f"validate data count: {len(val_x)}, {len(val_y)}")
    print(f"test data count: {len(test_x)}, {len(test_y)}")

    # test
    new_model = BLSTMCRFModel.load_model('./bert_chn_model')
    news = """「DeepMind 击败人类职业玩家的方式与他们声称的 AI 使命，以及所声称的『正确』方式完全相反。」
    DeepMind 的人工智能 AlphaStar 一战成名，击败两名人类职业选手。掌声和欢呼之余，它也引起了一些质疑。在前天 DeepMind 举办的 AMA 中，AlphaStar 项目领导者 Oriol Vinyals 和 David Silver、职业玩家 LiquidTLO 与 LiquidMaNa 回答了一些疑问。不过困惑依然存在……
    """
    print(f"news: {news}")
    res = new_model.predict(news, output_dict=True)
    print(f"res: {res}")


if __name__ == '__main__':
    # must run predict once before `app.run` to prevent predict error
    main()
