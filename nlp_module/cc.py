# -*- coding: utf8 -*-
# import opencc
from hanziconv import HanziConv


# def trans_t2s(text):
#     return opencc.convert(text)
#
#
# def trans_s2t(text):
#     # return opencc.convert(text, config='zhs2zht.ini').encode('utf8')
#     return opencc.convert(text, config='s2t.json')


def trans_t2s(text):
    return HanziConv.toSimplified(text)


def trans_s2t(text):
    return HanziConv.toTraditional(text)


if __name__ == '__main__':
    traditional_chinese_words = '嘉義縣水上鄉公所村幹事侯姓女子承辦選務，18日到印刷廠清點選票，離開時被警員發現她帶著300多張選票，立刻逮人。'
    simplified_chinese_words = '李友钟，男，1973年4月出生，汉族，籍贯重庆，全日制大学，法学学士、经济学学士，'

    t2sresult = trans_t2s(traditional_chinese_words)
    s2tresult = trans_s2t(simplified_chinese_words)
    #
    #
    print(t2sresult)
    print(s2tresult)

