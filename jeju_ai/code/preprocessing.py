"""모든 csv에 적용하는 공통적인 전처리 

read_any_csv -> clean_table -> 정리된 표

사용자가 올린 csv와 내장 csv를 동일한 처리
- 특정 데이터의 컬럼 이름을 코드에 쓰지 않는다. (날짜 '후보 이름' 목록만 예외: 이름 규칙이라서)
- 처리 순서: 글자로 읽기 → 보이지 않는 글자·NFC·깨진 글자 → 날짜 판별 → 숫자 변환
- report에는 '무엇을 바꿨는지'를 남긴다. 기획서·결과 설명자료에서 전처리 근거로 그대로 씀.
"""


import io 
import re
import unicodedata
from pathlib import Path


import pandas as pd

ENCODINGS = ["utf-8-sig","cp949"]

INVISIBLE_DELETE = "﻿​‌‍"
INVISIBLE_TO_SPACE = " 　"

BROKEN_MARKS = ["?", "�"] 
BROKEN_PATTERN = re.compile(
    r"(?<=[가-힣])[" + "".join(re.escape(m) for m in BROKEN_MARKS) + r"](?=[가-힣])"
)
MOJIBAKE_HINT = re.compile(r"[ÃìêÂ]")


DATE_EXACT = {
    "base_year_month", "year_month", "yearmonth", "ym", "yyyymm", "yymm", "base_ym",
    "stnd_ym", "stnd_yymm", "base_date", "base_ymd", "date",
    "년월", "연월", "기준년월", "기준연월", "날짜", "일자", "기준일", "기준일자",
}
DATE_LAST_PIECE = {"ym", "yymm", "yyyymm", "date", "ymd", "dt", "datetime"}
DATE_KO_CONTAINS = ["년월", "연월", "일자", "날짜", "기준일"]

# 값 길이 → 시도할 형식 [(format, 단위)] 
FORMATS_BY_LEN = {
    6: [("%Y%m", "월"), ("%Y-%m", "월"), ("%Y.%m", "월"), ("%Y/%m", "월")],
    7: [("%Y-%m", "월"), ("%Y.%m", "월"), ("%Y/%m", "월"), ("%Y년%m월", "월")],
    8: [("%Y%m%d", "일"), ("%Y-%m-%d", "일"), ("%Y.%m.%d", "일"), ("%Y/%m/%d", "일"),
        ("%Y년%m월", "월"), ("%Y년 %m월", "월")],
    9: [("%Y-%m-%d", "일"), ("%Y.%m.%d", "일"), ("%Y/%m/%d", "일"), ("%Y년%m월%d일", "일"),
        ("%Y년 %m월", "월")],
    10: [("%Y-%m-%d", "일"), ("%Y.%m.%d", "일"), ("%Y/%m/%d", "일"), ("%Y년%m월%d일", "일")],
    11: [("%Y.%m.%d.", "일"), ("%Y년%m월%d일", "일"), ("%Y년 %m월 %d일", "일")],
    12: [("%Y%m%d%H%M", "일시"), ("%Y년 %m월 %d일", "일")],
    13: [("%Y년 %m월 %d일", "일")],
    14: [("%Y%m%d%H%M%S", "일시")],
    16: [("%Y-%m-%d %H:%M", "일시"), ("%Y.%m.%d %H:%M", "일시"), ("%Y/%m/%d %H:%M", "일시")],
    19: [("%Y-%m-%d %H:%M:%S", "일시"), ("%Y.%m.%d %H:%M:%S", "일시"), ("%Y/%m/%d %H:%M:%S", "일시")],
}


ATE_PASS_RATIO = 0.95  # 빈 값을 뺀 값 중 95% 이상 변환되면 날짜로 확정
YEAR_RANGE = (1900, 2100)  # '500101' 같은 코드값이 5001년으로 통과하는 것을 막음
SAMPLE_N = 1000  # 판별은 일부만 보고, 확정 뒤에 전체 변환 (계산 절약)

NUMERIC_PASS_RATIO = 0.98  # 쉼표 뺀 뒤 98% 이상 숫자로 바뀌는 컬럼만 숫자로 확정
WEEKDAY_KO = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}

def read_any_csv(source) -> tuple[pd.DataFrame, str]:
    """파일이 저정된 경로를 받거나 또는 업로드된 파일 객체를 받아 '전부 글자 (str)'로 읽는다"""

    if isinstance(source,(str, Path)):
        raw = Path(source).read_bytes

    elif isinstance(source, bytes):
        raw = source

    else : #Streamit의 uploadFile같은 객체 
        raw = source.read()


    last_error = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(raw),dtype  = str , encoding = enc)
            return df,enc
        except UnicodeDecodeError as e:  #이 인코딩이 아닐 경우에는 다음 인코딩으로 
            last_error = e
    raise ValueError(f"CSV 인코딩을 할 수 없습니다 파일을 CSV 형태로 전달해주세요 ({last_error})")


def _clean_text(s : str) -> str: 
    """값 하나(또는 컬럼 이름 하나)의 보이지 않는 글자 정리 + NFC 처리"""


    for ch in INVISIBLE_DELETE:
        s = s.replace(ch, "")
    for ch in INVISIBLE_TO_SPACE:
        s = s.replace(ch, " ")
    return unicodedata.normalize("NFC", s).strip()


def _clean_text_column(col : pd.Series, name : str , report : dict) -> pd.Series : 

    """글자 컬럼 하나를 정리하고 바꾼 갯수를 report에 기록한다 """
    notna = col.notna()
    values = col[notna]

    uniques = pd.Series(values.unique())
    cleaned = uniques.map(_clean_text)
    fixed = cleaned.str.replace(BROKEN_PATTERN,"",regex= True)

    n_invisible = int((uniques != cleaned).sum())  #보이지 않는 오류 갯수 
    n_broken = int((cleaned != fixed).sum()) #오류난 글자 갯수 

    if n_invisible:
        report["invisible_fixed"][name] = n_invisible
    if n_broken:
        #어떤 값이 어떻게 바뀌었는지 몇개만 예시로 남긴다 (report가 너무 커지지 않게..)

        examples = [f"{a} -> {b}" for a,b in zip(cleaned[cleaned != fixed], fixed[cleaned != fixed])][:3]
        report["broken_fixed"][name] = {"count":n_broken,"examples": examples}

    if fixed.str.contains(MOJIBAKE_HINT).mean() > 0.3: 
        report["warnings"].append(f"'{name}'컬럼 글자가 깨져 보입니다 ,인코딩 확인 필요 ")

    if fixed.str.contains("[□■]").any():
        report["warnings"].append(f"'{name}' 컬럼에 □/■ 기호가 있습니다 (원래 값인지 확인 필요, 지우지 않음)")

    mapping = dict(zip(uniques, fixed))

    out = col.copy()
    out[notna] = values.map(mapping)

    return out