# encoding: utf-8
"""
@author: LIPan
@contact: panlee@qq.com
@version: 1.0
@time: 2019-05-24

"""
import os
import jieba
from kashgari.tasks.classification import CNNModel, BLSTMModel

FILEPATH = os.path.dirname(os.path.realpath(__file__))
FILENAME = os.path.basename(os.path.realpath(__file__))
BERT_CHN = 'D:\\ShareFolder\\_LIPan\\vobs\\_datasets\\nlp\\bert\\chinese_L-12_H-768_A-12'


def main():
    new_model = BLSTMModel.load_model(os.path.join(FILEPATH + '/models/thucnews_bert_model'))
    news = """「DeepMind 击败人类职业玩家的方式与他们声称的 AI 使命，以及所声称的『正确』方式完全相反。」
    DeepMind 的人工智能 AlphaStar 一战成名，击败两名人类职业选手。掌声和欢呼之余，它也引起了一些质疑。在前天 DeepMind 举办的 AMA 中，AlphaStar 项目领导者 Oriol Vinyals 和 David Silver、职业玩家 LiquidTLO 与 LiquidMaNa 回答了一些疑问。不过困惑依然存在……
    """
    news1 = """ 一段时间以来，部分美国政客一再制造关于华为的谣言，但始终拿不出各国要求提供的证据。美国国内也对美方挑起贸易战、科技战造成的市场动荡、产业合作受阻发出越来越多质疑。于是，这些美国政客不断编造各种主观推定的谎言试图误导美国民众，现在又试图煽动意识形态对立。但这是不合逻辑的，放眼世界，意识形态分歧并不必然妨碍国家间的经贸、产业、科技合作。中美建交40年来，美国两党历届政府不正是一直同中国共产党领导的中国政府一道持续推进、拓展、深化各领域互利合作吗？事实上，中美建交之初两国签署的首批政府间合作协议就包括了《中美科技合作协定》。另一方面，棱镜门事件也好、阿尔斯通事件也好，都让世人看清，所谓意识形态的趋同，也并没有妨碍美国对自己的盟友采取各种不正当的手段。
    
    """
    x = list(jieba.cut(news))
    print(f'x:{len(x)}')
    res = new_model.predict(x)
    print(f"res: {res}")

    x1 = list(jieba.cut(news1))
    print(f'x1:{len(x1)}')
    res1 = new_model.predict(x1)
    print(f"res1: {res1}")


if __name__ == '__main__':
    # must run predict once before `app.run` to prevent predict error
    main()
