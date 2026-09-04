import enchant
import re
from nltk.stem.wordnet import WordNetLemmatizer
from itertools import groupby
import xlrd

special_words = ['prosses', 'acsses']


# 出去一个字符中不恰当的符号和其他的一些成分，具体见下面的代码。处理后的字符串中含有的token数量小于200输出false
def clean_text(input_text):
    d = enchant.Dict('en_US')
    input_text = input_text.lower()
    input_text = re.sub(r'[-+\n]', '', input_text)  # 把整个句子的-+换行替换为空格
    input_text = re.sub(r'[~;]+', '.', input_text)  # 把整个句子中连续多个~或者; 合为.
    input_text = re.sub(r"[']+", r"'", input_text)  # 把整个句子中连续多个' 合为一个'
    input_text = re.sub(r"[^0-9a-z,.?!<>' ]", ' ', input_text)  # 把整个句子中除了数字、字母、句号、逗号、问号、感叹号，尖角号之外的所有符号全部转为空字符
    input_text = re.sub(r"([,.?!])\1+", r' \1 ', input_text)  # 把整个句子中连续出现的,.?!合为一个（空格）符号（空格）
    input_text = re.sub(r"([,.?!])", r' \1 ', input_text)  # 把整个句子中每一个,.?!替换为（空格）符号（空格）
    input_text = re.sub(r'[ ]+', ' ', input_text)  # 把整个句子中连续的空格合为一个空格
    if re.search(r'\w$', input_text):  # 如果字符串末尾使用数字、字母、下划线就给源字符串加上一个（空格）句号（空格）
        input_text += ' . '
    # token = nltk.word_tokenize(rows[i][review_no])
    token = input_text.split()  # 分离单词和标点符号
    for word_no in range(len(token)):
        token[word_no] = re.sub(r"^'|'$", r'', token[word_no])  # 扫描这个token开头是不是'，是的话消除，不是的话再看末尾是不是'，是的话消除。
        token[word_no] = re.sub(r'(.)\1{2,}', r'\1', token[word_no])  # 扫描整个token，把连续(频率为2次以上)出现的..转为一个.
        # token[word_no] = WordNetLemmatizer().lemmatize(token[word_no], 'n')
        # token[word_no] = WordNetLemmatizer().lemmatize(token[word_no], 'v')
        token[word_no] = WordNetLemmatizer().lemmatize(token[word_no], 'v')  # 默认为动词，进行词形还原
        if token[word_no] not in special_words:
            temp_token = WordNetLemmatizer().lemmatize(token[word_no], 'n')  # 如果token不在special_words里面，按照名词再次还原
            if (len(token[word_no]) > 3 and token[word_no] != temp_token and not re.search(r'ss$', token[
                word_no])):  # 如果单词长度大于3，并且两次还原有差异，同时不是ss结尾
                # lemmatize_lis.add((token[word_no], temp_token))
                token[word_no] = temp_token

    # remove consecutive words
    token = [x[0] for x in groupby(token)]  # 连续出现完全相同的token只取一个

    # remove long word
    words = list(token)
    output_text = ' '.join(words)
    output_text = re.sub(r'[ ]+', ' ', output_text)  # 扫描整个字符串，连续出现的空格转为一个空格
    output_text = re.sub(r'(. )\1+', r'\1', output_text).strip()  # 扫描整个字符串，连续出现的.（空格）,转为一次.(空格)
    length = len(output_text.split(" "))
    if re.search(r'[a-zA-Z]', output_text) and length <= 200:  # 如果字符串含有一个字母，同时token的个数小于等于200，认为合格
        return output_text
    else:
        print(length)
        print(output_text.split(" "))
        return False


def description_generate(original_file, target_file):
    workbook = xlrd.open_workbook(original_file)
    booksheet = workbook.sheet_by_index(0)
    app_id = []
    description_original = []
    description_processed = []
    for i in range(booksheet.nrows - 1):
        app_id.append(booksheet.cell_value(i + 1, 0))
        description_original.append(booksheet.cell_value(i + 1, 1))
    for t in description_original:
        description_processed.append(clean_text(t))

    with open(target_file, 'wt', encoding='utf-8', newline='') as f:
        for i in range(len(app_id)):
            f.write(app_id[i] + "-[split]-" + description_processed[i] + "\n")