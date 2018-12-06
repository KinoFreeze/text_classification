# coding:utf-8
import sys
import jieba
reload(sys)

sys.setdefaultencoding('utf-8')



def read_file(read_path):
    temp = open(read_path, "r")
    content = temp.readlines()
    temp.close()
    return content


def text_divide(content):
    token_list = []
    list_ =[]
    for row in content:
        print(row)
        items = row.strip().split("	", 1)
        items_list = list(jieba.cut(items[0]))
        for temp in items_list:
            if ' ' in items_list:
                items_list.remove(' ')
            if len(temp) == 1:
                list_.append(temp + "\t" + "S")
            elif len(temp) == 2:
                j = 1
                for i in temp:
                    if j == 1:
                        list_.append(i + "\t" + "B" + "\n")
                        j = 2
                    elif j == 2:
                        list_.append(i + "\t" + "E")
            elif (len(temp) != 1) and (len(temp) != 2):
                j = 1
                for i in temp:
                    if j == 1:
                        list_.append(i + "\t" + "B" + "\n")
                        j += 1
                    elif j != len(temp):
                        list_.append(i + "\t" + "M" + "\n")
                        j += 1
                    else:
                        list_.append(i + "\t" + "E")
            list_.append("\n")
        list_.append("\n")
    return list_


def write_file(write_path, content):
    temp = open(write_path, "w")
    temp.write("".join(content))
    temp.close()


if __name__ == "__main__":
    content = text_divide(read_file("/home/wzc/data/nlpcc2016/nlpcc2016-word-seg-train.dat"))
    write_file("/home/wzc/data/nlpcc2016/devided/nlpcc2016-word-seg-train-devided.txt", content)
