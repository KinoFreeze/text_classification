# coding:utf-8
import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def read_file(read_path):
    read = open(read_path, "r")
    content = read.readlines()
    read.close()
    return content


def write_file(content, write_path):
    write = open(write_path, "w")
    write.write("".join(content))
    write.close()


def devided_text(content, dictionary_path):
    jieba.load_userdict(dictionary_path)
    result = []
    for row in content:
        items = row.strip().split(" ",1)
        tokens = list(jieba.cut(items[0]))
        token = " ".join(tokens)
        result.append(token)
    return result



if __name__ == "__main__":
    content = read_file("/home/wzc/data/nlpcc2016/wzc.txt")
    result = devided_text(content, "/home/wzc/data/nlpcc2016/hlt_stop_words.txt")
    write_file(result, "/home/wzc/data/nlpcc2016/devided/nlpcc2016-wordseg-test-devided.txt")