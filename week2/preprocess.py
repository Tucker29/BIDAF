import numpy as np
import data_io as pio
[['aaa','bbbb'],['ccc','ddd']]
[[['a','a,'a'],['b','b,'b','b']],[['c','c','c'],['d','d,'d']]]

import nltk

tmp_path0="D:/dhm/programer-lx/BiDAF_tf2"
GLOVE_FILE_PATH="tmp_path0+"/data/glove/glove.6B.50d.txt"

class Preprocessor:

    def __init__(self, datasets_fp, max_length=384, stride=128):
        self.datasets_fp = datasets_fp
        self.max_length = max_length
        self.max_clen = 100
        self.max_qlen = 100
        self.max_char_len = 16
        self.stride = stride
        self.charset = set()
        self.wordset = set()
        self.embeddings_index={}
        self.embeddings_matrix=[]
        self.load_glove(GLOVE_FILE_PATH)	#word_list来自于glove
        self.build_charset_wordset()    # 01 建立字符集  & 词典集

    def build_charset_wordset(self):
        # 01 建立字符集 & 词典集
        for fp in self.datasets_fp: # 3个文件
            print("fp=",fp)
            self.charset, self.wordset |= self.dataset_info(fp)   # 02

        self.charset = sorted(list(self.charset))   # 对字符集进行排序
        self.charset = ['[PAD]', '[CLS]', '[SEP]'] + self.charset + ['[UNK]']
        idx = list(range(len(self.charset)))        # 根据字符集的大小，建立索引数
        self.ch2id = dict(zip(self.charset, idx))   # 字符 索引
        self.id2ch = dict(zip(idx, self.charset))   # 索引 字符

        idx = list(range(len(self.wordset)))        # 根据词集的大小，建立索引数
        self.w2id = dict(zip(self.wordset, idx))   # 词 索引
        self.id2w = dict(zip(idx, self.wordset))   # 索引 词


    def dataset_info(self, inn):
        # 02 获取文件中内容、问题、答案字段中的字集 & 词集

        charset = set()             # 初始化字符集变量
        wordset = set()             # 初始化词集变量

        dataset = pio.load(inn)     # data_io

        for _, context, question, answer, _ in self.iter_cqa(dataset):  # 03 获取qid, context, question, answer, answer_start中的context, question, answer
            charset |= set(context) | set(question) | set(answer)   # 并集运算
            # self.max_clen = max(self.max_clen, len(context))
            # self.max_qlen =

            #rec_text=new_word_tokenize(" "+context+" "+question+" "+answer+" ").lower().split()
            #wordset |= set(rec_text)
            wordset |= set(context) | set(question) | set(answer)   # 并集运算

        return charset,wordset

    def iter_cqa(self, dataset):
        # 03 从文本中剥离从所需字段：qid, context, question, text, answer_start

        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        yield qid, context, question, text,answer_start
'''
尝试改进seg_text中，清除符号部分（待继续）
    def new_word_tokenize(text_str1,text_str12):
        # 清理文中的标点符号等，待进一步完善
        for st in r_str1:
            if st + " " in text_str:
                st += " "
                text_str = text_str.replace(st, ' ')

        for st in l_str1:
            if " " + st in text_str:
                st = " " + st
                text_str = text_str.replace(st, ' ')

        # , "'ll", "'m"
        for st in kepp_str:
            st += " "
            text_str = text_str.replace(st, ' ' + st)

        return text_str
'''

    def char_word_encode(self, context, question):
        q_seg_list = self.seg.text(question)
        c_seg_list = self.seg.text(context)

        question_encode = self.convert2id_char(word_list=q_seg_list, max_char_len=self.max_char_len, begin=True,
                                               end=True)  # 05
        context_encode = self.convert2id_char(word_list=c_seg_list, max_char_len=self.max_char_len,
                                              maxlen=self.max_length - len(question_encode), end=True)  # 05
        ccq_encode = question_encode + context_encode

        question_encode = self.convert2id_word(word_list=q_seg_list, begin=True, end=True)  # 05
        context_encode = self.convert2id_word(word_list=c_seg_list, maxlen=self.max_length - len(question_encode),
                                              end=True)  # 05
        wcq_encode = question_encode + context_encode

        assert len(ccq_encode) == self.max_length
        assert len(wcq_encode) == self.max_length

        return ccq_encode, wcq_encode

    def convert2id_char(self, word_list=[], max_clen = None, maxlen = None, begin = False, end = False):
        # 05
        ##ch = [ch for ch in sent]
        ##ch = ['[CLS]'] * begin + ch

        char_list = []
        char_list = [[self.get_id_char(['[CLS]')] + [self.get_id_char('[PAD]')] + (max_char_len - 1 * begin + char_list)

        for word in word_list:
            ch =[ch for ch in word]
            if max_char_len is not None:
                ch = ch[:max_char_len]

            ids = list(map(self.get_id_char, ch))  # 06
            while len(ids) < max_char_len:
                ids.append(self.get_id_char('[PAD]')
                char_list.append(np.array(ids))

        if maxlen is not None:
            char_list = char_list[:maxlen - 1 * end]
            char_list += [[self.get_id_char('[PAD]')] * max_char_len] * (maxlen - len(char_list))

        return char_list

    def convert2id_word(self, word_list=[], maxlen=None, begin=False, end=False):
        # 05
        ch = [ch for ch in word_list]
        ch = ['CLS'] * begin + ch

        if maxlen is not None:
            ch = ch[:maxlen - 1 * end]
            ch += ['PAD'] * (maxlen - len(ch))

        ids = list(map(self.get_id_word, ch))  # 06

        return ids

    def get_id_char(self, ch):
        # 06
        return self.ch2id.get(ch, self.ch2id['[UNK]'])

    def get_id_word(self, ch):
        # 06'
        return self.w2id.get(ch, self.w2id['unk'])

    def get_dataset(self, ds_fp):
        # 07 从数据集中抽取c,q,b,e--->从数据集中抽取ccs, qcs, cws, qws,b,e
        ##cs, qs, be = [], [], []
        ccs, qcs, cws, qws, be = [], [], [], [], []
        ##for _, c, q, b, e in self.get_data(ds_fp):
        for _, cc, cq, wc, wq, b, e in self.get_data(ds_fp):
            ccs.append(cc)
            qcs.append(cq)
            cws.append(cw)  #新增
            qws.append(qw)  #新增
            be.append((b, e))
        ##return map(np.array, (cs, qs, be))
        return map(np.array, (ccs, qcs, cwc, qws, be))

    def get_data(self, ds_fp):
        # 08 从数据集中分类qid, context, question, text, answer_start
        dataset = pio.load(ds_fp)   # 调用data_io加载数据

        for qid, context, question, text, answer_start in self.iter_cqa(dataset):   # 调用03
            c_seg_list,q_seg_list = self.seg_text(context,question)
            #q_seg_list = self.seg_text(question)
            ##cids = self.get_sent_ids(context, self.max_clen)  # 调用09
            ##qids = self.get_sent_ids(question, self.max_qlen)

            c_char_ids = self.get_sent_ids_char(self.max_clen=max_clen, word_list=c_seg_list)            # 调用09
            q_char_ids = self.get_sent_ids_char(self.max_qlen=max_qlen, begin=True, word_list=q_seg_list)
            c_word_ids = self.get_sent_ids_word(self.max_clen=max_clen, word_list=c_seg_list)            # 调用09【新增】
            q_word_ids = self.get_sent_ids_word(self.max_qlen=max_qlen, begin=True, word_list=q_seg_list)
            b, e = answer_start, answer_start + len(text)
            nb, ne, len_all_char = -1, -1, 0
            ##if e >= len(cids):
            ##    b = e = 0
            ##yield qid, cids, qids, b, e
            # 对于长度的处理
            for i, w in enumerate(c_seg_list):
                if i == 0:
                    contiune
                if b > len_all_char - 1 and b <= len_all_char + len(w) - 1:
                    b = i + 1
                if e > len_all_char - 1 and e <= len_all_char + len(w) - 1:
                    e = i + 1
                len_all_char += len(w)
            if ne == -1:
                b = e = 0
            yield qid, c_char_ids, q_char_ids,c_word_ids, q_word_ids,b, e

    def get_sent_ids_char(self, maxlen=0,begin=False,end=True,word_list=[]):
        # 09 建立句子的索引
        return self.convert2id_char(word_list=self.char_list, max_char_len=self.max_char_len,maxlen=maxlen, begin=False,end=True)   # 调用函数05

    def get_sent_ids_word(self, maxlen=0,begin=False,end=True,word_list=[]):
        # 09'
        return self.convert2id_word(word_list=self.word_list,maxlen=maxlen, begin=False,end=True)  # 调用函数05

    def seg_text(self, text1,text2):
        words = [word.lower() for word in nltk.word_tokenize(text1)]
        words1 = [word.lower() for word in nltk.word_tokenize(text2)]
        reture words,words1

    def load_glove(self, glove_file_path):
        # 每一行为：单词，向量
        with open(glove_file_path, 'r', encoding='utf-8') as f
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, sep=' ')
                self.embeddings_index[word] = coefs
                self.word_list_index.append(word)
                self.embedding_matrix.append(coefs)

if __name__ == '__main__'
    # 实例化Preprocessor
    p = Preprocessor([
        tmp_path+'/data/squad/train-v1.1.json',
        tmp_path+'/data/squad/dev-v1.1.json',
        tmp_path+'/data/squad/dev-v1.1.json'
    ])

    print(p.char_word_encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))  # 调用04函数encode
    #print(p.word_encode('modern stone statue of Mary', 'To whom did the Virgin Mary '))  # 调用04函数encode
else:   #观察一下__name__
    print("__name__###########",__name__)
