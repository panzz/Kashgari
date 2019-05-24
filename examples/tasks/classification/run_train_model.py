
# encoding: utf-8
"""
@author: LIPan
@contact: panlee@qq.com
@version: 1.0
@time: 2019-05-24

"""
import os
import jieba
import tqdm

BERT_CHN = 'D:\\ShareFolder\\_LIPan\\vobs\\_datasets\\nlp\\bert\\chinese_L-12_H-768_A-12'
THUCNEWS = 'D:\\ShareFolder\\_LIPan\\vobs\\_datasets\\nlp\\'

FILENAMES = {
    'train': 'THUCNews/cnews.train.txt',
    'valid': 'THUCNews/cnews.val.txt',
    'test': 'THUCNews/cnews.test.txt',
}

def read_data_file(path):
    lines = open(path, 'r', encoding='utf-8').read().splitlines()
    x_list = []
    y_list = []
    for line in tqdm.tqdm(lines):
        rows = line.split('\t')
        if len(rows) >= 2:
            y_list.append(rows[0])
            x_list.append(list(jieba.cut('\t'.join(rows[1:]))))
        else:
            print(rows)
    return x_list, y_list


def main(): 
    test_x, test_y = read_data_file(os.path.join(THUCNEWS, FILENAMES['test']))
    val_x, val_y = read_data_file(os.path.join(THUCNEWS, FILENAMES['valid']))
    train_x, train_y = read_data_file(os.path.join(THUCNEWS, FILENAMES['train']))
    # train_x, train_y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(
    #     'train')
    # val_x, val_y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(
    #     'validate')
    # test_x, test_y = ChinaPeoplesDailyNerCorpus.get_sequence_tagging_data(
    #     'test')

    print(f"train data count: {len(train_x)}, {len(train_y)}")
    print(f"validate data count: {len(val_x)}, {len(val_y)}")
    print(f"test data count: {len(test_x)}, {len(test_y)}")

    # embedding = BERTEmbedding(BERT_CHN, 200)
    # # 还可以选择 `BLSTMModel` 和 `CNNLSTMModel`
    # tf_board_callback = keras.callbacks.TensorBoard(
    #     log_dir='./logs', update_freq=1000)
    # model = BLSTMCRFModel(embedding)
    # res = model.fit(train_x,
    #                 train_y,
    #                 y_validate=val_y,
    #                 x_validate=val_x,
    #                 epochs=1,
    #                 batch_size=500,
    #                 fit_kwargs={'callbacks': [tf_board_callback]})
    # print(f"res: {res}")
    # model.evaluate(test_x, test_y)
    # model.save('./bert_chn_model')

    # # test
    # new_model = BLSTMCRFModel.load_model('./bert_chn_model')
    # news = """「DeepMind 击败人类职业玩家的方式与他们声称的 AI 使命，以及所声称的『正确』方式完全相反。」
    # DeepMind 的人工智能 AlphaStar 一战成名，击败两名人类职业选手。掌声和欢呼之余，它也引起了一些质疑。在前天 DeepMind 举办的 AMA 中，AlphaStar 项目领导者 Oriol Vinyals 和 David Silver、职业玩家 LiquidTLO 与 LiquidMaNa 回答了一些疑问。不过困惑依然存在……
    # """
    # print(f"news: {news}")
    # res = new_model.predict(news, output_dict=True)
    # print(f"res: {res}")


if __name__ == '__main__':
    # must run predict once before `app.run` to prevent predict error
    main()
