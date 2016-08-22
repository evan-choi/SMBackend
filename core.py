#coding: utf-8

import re
from konlpy.tag import Twitter; t = Twitter()
from collections import deque
from itertools import islice
import nltk

swords = []#["삼성", "BC"]

class Core:
    @staticmethod
    def getTwitter():
        return t

    @staticmethod
    def ngrams(message, n=1):
        it = iter(message.split())
        window = deque(islice(it, n), maxlen=n)
        yield tuple(window)
        for item in it:
            window.append(item)
            yield tuple(window)

    @staticmethod
    def analyze(d, n=5):
        global swords

        # n = " ".join([x[0] for y in word_tokenize(d) if pos_tag([y])[0][1][:2] in check_en for x in t.pos(y) if x[1] in check])

        # |상품|해외|카드|국산|발매|가입비|약정|최저가|대행|사은품|판매|회원|주문|입니다|영수증|중고|습니다|감사합니|출고|증정|직구
        # d = re.sub(u"((?:\d+,?)+만?원|\d+%|(?:[가-힣]*(?:공식|만족|최우수|매출\d?위?|무료|배송)[가-힣]*))", '', d)
        ename = re.sub(u"[^가-힣A-Za-z0-9-_]+", ' ', d)
        nname = re.sub(u"[^A-Za-z0-9-_]+", ' ', d)
        ename = re.sub(u" +", ' ', ename).strip()

        result = list(set(nltk.word_tokenize(ename)).union(x[0] for x in t.pos(ename)))
        #result = list(set(nltk.word_tokenize(nname)).union(x for x in t.nouns(ename)))
        # result = [x[0] for x in t.pos(ename)]

        if ename.find(' ') != -1:
            for x in Core.ngrams(ename, n):
                result.append(x[0] + x[1])

        rr = []
        for r in result:
            if len(r) > 1:
                rr.append(r)

        for w in swords:
            if w in rr:
                rr.remove(w)

        return list(set(rr))

    @staticmethod
    def proc(d, n=5):
        r = " ".join(Core.analyze(d, n))
        return r
