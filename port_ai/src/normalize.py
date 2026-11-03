"""
모듈 ① 정규화 + 한글 발음 키
"""

from os import name
import unicodedata
import re
from hangulpy import Romanizer

    
#제 → (삭제), 호 → (삭제),NO ->(삭제)
preprocess_word = (
    '제','호','NO',"no",
)

#이름을 정규화 후 리스트로 반환
def normalize_name(value: object) -> list[str]:
    if not isinstance(value, str):          # 문자열이 아니면(빈 칸 NaN 등) 빈 문자열
        return ""
    value1 = unicodedata.normalize("NFKC", value).upper()   # 글자 통일 + 대문자
    value2 = "".join(ch for ch in value1 if ch.isalnum() or ch == " ")  # 글자·숫자,공백만 남기고 이어붙이기
    value_list = [i.strip() for i in re.findall(r"\d+|[A-Z]+|[가-힣]+",value2)]

    for i in range(len(value_list),0):
        if (value_list[i] in preprocess_word):
            del value_list[i] #preprocess_word에 속하는 리스트 원소는 삭제하기 

    return value_list


if __name__ == "__main__":
    print(normalize_name("제 7 하나"))
    print(normalize_name("NO.7 HANA"))

#정확도를 위해 여러개의 모델을 만들고 단 한번만 만든다 
_ROMANIZERS = (
    Romanizer("revised", mode="general"),  # 현행 표기
    Romanizer("revised", mode="proper"),   # 고유명사 ㄱㄷㅂ+ㅎ (묵호 → mukho)
    Romanizer("mr"),                       # 옛 표기 (부산 → pusan)
)
#한국 단어를 로마자 발음 표기로 바꾸며,정규화를 진행한 단어만 들어온다 (리스트가 들어오지 않는다)
def ko_phon(word) -> set :
    return list({fold(r.romanize(word).replace("eu", "")) for r in _ROMANIZERS})




#영어 단어를 로마자 발음 표기로 바꾸며,정규화를 진행한 단어만 들어온다 
def eng_phon(word) -> str:
    return fold(word)


def fold(s) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("’", "") 
    for a, b in _FOLD:
        s = s.replace(a, b)
    return s

_FOLD = (("eo", "o"), ("eu", "u"), ("ch", "c"), ("j", "c"),
         ("g", "k"), ("d", "t"), ("b", "p"))

def is_hangul_name(value: object) -> bool:
    return isinstance(value, str) and any("가" <= char <= "힣" for char in value)