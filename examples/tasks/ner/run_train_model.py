
# encoding: utf-8
"""
@author: LIPan
@contact: panlee@qq.com
@version: 1.0
@time: 2019-05-24

"""
import os
import random
import keras
# from kashgari.corpus import ChinaPeoplesDailyNerCorpus
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.seq_labeling import BLSTMCRFModel
from kashgari.utils import helper
from kashgari.utils.logger import init_logger
init_logger()

# define where you are
filePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
fileName = os.path.basename(os.path.realpath(__file__))

BERT_CHN = 'D:\\ShareFolder\\_LIPan\\vobs\\_datasets\\nlp\\bert\\chinese_L-12_H-768_A-12'
BERT_UNCASEED_L12_H768_A12 = 'D:\\ShareFolder\\_LIPan\\vobs\\_datasets\\nlp\\bert\\uncased_L-12_H-768_A-12'
BERT_MULTI = 'D:\\ShareFolder\\_LIPan\\vobs\\_datasets\\nlp\\bert\\multi_cased_L-12_H-768_A-12'

model_path = os.path.join('models','ber_bert_model_uncased_L12H768_unfinetune')


DATA_TRAIN = 'train'
DATA_VALIDATE = 'validate'
DATA_TEST = 'test'


def _load_data_and_labels(filename, encoding='utf-8'):
    """Loads data and label from a file.
    Args:
        filename (str): path to the file.
        encoding (str): file encoding format.
        The file format is tab-separated values.
        A blank line is required at the end of a sentence.
        For example:
        ```
        EU	B-ORG
        rejects	O
        German	B-MISC
        call	O
        to	O
        boycott	O
        British	B-MISC
        lamb	O
        .	O
        Peter	B-PER
        Blackburn	I-PER
        ...
        ```
    Returns:
        tuple(numpy array, numpy array): data and labels.
    Example:
        >>> filename = 'conll2003/en/ner/train.txt'
        >>> data, labels = load_data_and_labels(filename)
    """
    sents, labels = [], []
    words, tags = [], []
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []

    return sents, labels

class CoNLL2003Corpus(Corpus):
    __corpus_name__ = 'corpus/conll2003'
    __zip_file__name = 'corpus/conll2003.tar.gz'

    @classmethod
    def get_sequence_tagging_data(cls,
                                  data_type: str = DATA_TRAIN,
                                  folder_path: str = os.path.join('datasets', 'conll2003'),
                                  task_name: str = 'ner',
                                  shuffle: bool = True,
                                  max_count: int = 0) -> Tuple[List[List[str]], List[List[str]]]:
        # folder_path = helper.cached_path(cls.__corpus_name__, cls.__zip_file__name)
        
        print(f"CoNLL2003Corpus> init folder_path:{folder_path}")
        if data_type not in [DATA_TRAIN, DATA_VALIDATE, DATA_TEST]:
            raise ValueError('data_type error, please use one onf the {}'.format([DATA_TRAIN,
                                                                                  DATA_VALIDATE,
                                                                                  DATA_TEST]))
        if task_name not in ['ner', 'pos', 'chunking']:
            raise ValueError('data_type error, please use one onf the {}'.format(
                ['ner', 'pos', 'chunking']))
        folder_path = os.path.join(folder_path, task_name)
        print(f"CoNLL2003Corpus> add task folder_path:{folder_path}")
        if data_type == DATA_TRAIN:
            file_path = os.path.join(folder_path, 'train.txt')
        elif data_type == DATA_TEST:
            file_path = os.path.join(folder_path, 'test.txt')
        else:
            file_path = os.path.join(folder_path, 'valid.txt')
        x_list, y_list = _load_data_and_labels(file_path)
        if shuffle:
            x_list, y_list = helper.unison_shuffled_copies(x_list, y_list)
        if max_count:
            x_list = x_list[:max_count]
            y_list = y_list[:max_count]
        return x_list, y_list

    __desc__ = """
        http://ir.hit.edu.cn/smp2017ecdt-data
        """

def main():
    train_x, train_y = CoNLL2003Corpus.get_sequence_tagging_data(data_type='train', folder_path='')
    val_x, val_y = CoNLL2003Corpus.get_sequence_tagging_data(data_type='validate', folder_path='')
    test_x, test_y = CoNLL2003Corpus.get_sequence_tagging_data(data_type='test', folder_path='')

    print(f"train data count: {len(train_x)}, {len(train_y)}")
    print(f"validate data count: {len(val_x)}, {len(val_y)}")
    print(f"test data count: {len(test_x)}, {len(test_y)}")


    embedding = BERTEmbedding(BERT_UNCASEED_L12_H768_A12, 200)
    # 还可以选择 `BLSTMModel` 和 `CNNLSTMModel`
    tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)
    model = BLSTMCRFModel(embedding)
    res = model.fit(x_train=train_x,
                    y_train=train_y,
                    x_validate=val_x,
                    y_validate=val_y,
                    epochs=1,
                    batch_size=100,
                    fit_kwargs={'callbacks': [tf_board_callback]})
    print(f"res: {res}")
    model.evaluate(test_x, test_y)
    model.save(os.path.join(filePath, model_path))


if __name__ == '__main__':
    # must run predict once before `app.run` to prevent predict error
    main()
