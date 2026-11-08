import pandas as pd
from pathlib import Path
import re

MAIN = Path(__file__).parent.parent / "data" 



df = pd.read_csv(f"{MAIN}/economy_utf8/업종별성별카드이용정보.csv",encoding="utf-8-sig",dtype=str)



def prepare_data(df : pd.DataFrame) -> pd.DataFrame : 
    """① 날짜: base_year_month → year, month"""
    BROKEN_CHARS = "?\ufffd" 
    df.info()
    df.columns = df.columns.str.strip() # 컬럼 이름의 공백 제거
    df.columns = df.columns.str.lower() # 컬럼 이름의 공백 제거

    text_cols = df.select_dtypes(include=["object", "string"]).columns

    pattern = rf"(?<=[가-힣])[{re.escape(BROKEN_CHARS)}](?=[가-힣])"

    #컬럼 내용 전처리
    for col in text_cols:
        df[col] = df[col].strip()
        df[col] = df[col].replace(r"[\ufeff\u200b-\u200d]", "", regex=True) 
        df[col] = df[col].replace(r"[\xa0\u3000]", " ", regex=True)
        df[col] = df[col].str.normalize("NFC")
        df[col] = df[col].str.replace(pattern, "", regex=True)  # "기타음?식료품" → "기타음식료품"

        #df[col].str.contains(pattern, regex=True).sum()  # 고쳐질 값이 몇 개인지 (report용)

    
    #날짜  전처리

    # (가) 이 이름과 '정확히' 같으면 날짜 후보 (소문자 기준)
    EXACT_NAMES = {
    "base_year_month", "year_month", "yearmonth", "ym", "yyyymm", "yymm",
    "base_ym", "stnd_ym", "stnd_yymm",
    "base_date", "base_ymd", "date",
    }


    # (나) '_'(또는 공백, -)로 나눈 '마지막 조각'이 이 중 하나면 날짜 후보
    #      예: open_date → ["open", "date"] → 마지막 조각 "date" → 후보
    #      부분 문자열 검사를 하지 않는 이유: "payment" 안에 "ym"이 들어 있음
    SUFFIX_TOKENS = {"ym", "yymm", "yyyymm", "date", "ymd", "dt", "datetime"}
    
    # (다) 한글 컬럼명
    KOREAN_EXACT = {"년월", "연월", "기준년월", "기준연월", "날짜", "일자", "기준일", "기준일자"}
    KOREAN_CONTAINS = ["년월", "연월", "일자", "날짜", "기준일"]  # 예: "승차일자", "판매일자"

    date = []
    cols = [str(col) for col in df.columns]
    for col in cols:#중첩된 for문으로 해결할 것
        if col in EXACT_NAMES:
            date.append(col)
        
        elif re.split(r"[_\s-]+", col)[-1]  in SUFFIX_TOKENS :
            date.append(col)

        elif col in KOREAN_EXACT:
            date.append(col)

        else:
            for kor in KOREAN_CONTAINS:
                if kor in col:
                    date.append(col)
                    break

    M, D, T = "month", "day", "datetime"   # 단위 표시 (월 / 일 / 일시)

# 값 길이 → 그 길이에서 나올 수 있는 (형식, 단위) 목록
# 한 길이에 형식이 여러 개 → 전부 시도해서 성공 비율이 가장 높은 것을 고름
    FORMATS_BY_LEN = {
        6:  [("%Y%m", M), ("%Y-%m", M), ("%Y.%m", M), ("%Y/%m", M)],          # 201801, 2018-1
        7:  [("%Y-%m", M), ("%Y.%m", M), ("%Y/%m", M), ("%Y년%m월", M)],      # 2018-01, 2018년1월
        8:  [("%Y%m%d", D), ("%Y-%m-%d", D), ("%Y.%m.%d", D), ("%Y/%m/%d", D),
            ("%Y년%m월", M), ("%Y년 %m월", M)],                                # 20180115, 2018-1-5, 2018년01월
        9:  [("%Y-%m-%d", D), ("%Y.%m.%d", D), ("%Y/%m/%d", D),
            ("%Y년 %m월", M), ("%Y년%m월%d일", D)],                            # 2018-1-15, 2018년 01월
        10: [("%Y-%m-%d", D), ("%Y.%m.%d", D), ("%Y/%m/%d", D), ("%Y년%m월%d일", D)],  # 2018-01-15
        11: [("%Y.%m.%d.", D), ("%Y년%m월%d일", D), ("%Y년 %m월 %d일", D)],     # 2018.01.15.
        12: [("%Y%m%d%H%M", T), ("%Y년 %m월 %d일", D)],                       # 201801151639
        13: [("%Y년 %m월 %d일", D)],                                           # 2018년 01월 15일
        14: [("%Y%m%d%H%M%S", T)],                                             # 20180115163956
        16: [("%Y-%m-%d %H:%M", T), ("%Y.%m.%d %H:%M", T), ("%Y/%m/%d %H:%M", T)],
        19: [("%Y-%m-%d %H:%M:%S", T), ("%Y.%m.%d %H:%M:%S", T), ("%Y/%m/%d %H:%M:%S", T)],
    }
        
    if not date:
        print("표에 날짜 형식이 존재하지 않음") 
    else:
        for date_cadidate in date:
            head_date = df[date_cadidate].head(1000)
            head_date = head_date.dropna()
            date_len = head_date.str.len().mode()[0]
            date_format_candidate_list = FORMATS_BY_LEN[date_len]
            date_index = -1

            for i,format in enumerate(date_format_candidate_list):
                d = pd.to_datetime(head_date, format=format[0], errors="coerce")
                valid = d.dt.year.between(1900, 2100)
                accuracy_rate = valid.mean()        # True 비율 = 통과율 (True=1, False=0으로 평균)
                #fail_count = (~valid).sum()  # 실패한 값 개수
                if accuracy_rate >= 0.95 :
                    print(f"성공 : {accuracy_rate}")
                    date_index = i

            if date_index == -1:
                print(f"{date_cadidate} 컬럼은 날짜 관련 열이 아닙니다")
            else:
                 df[date_cadidate] = pd.to_datetime(df[date_cadidate],format = FORMATS_BY_LEN[date_len][date_index][0],errors="coerce") 
                 print(f"저장 형식 : {FORMATS_BY_LEN[date_len][date_index][1]}")

            

    df.info()




    return df

print(prepare_data(df))


