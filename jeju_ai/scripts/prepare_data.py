import pandas as pd
from pathlib import Path

MAIN = Path(__file__).parent.parent / "data" 

df = pd.read_csv(f"{MAIN}/economy_utf8/업종별성별카드이용정보.csv",encoding="utf-8-sig")

def prepare_data(df : pd.DataFrame) -> pd.DataFrame : 
    """① 날짜: base_year_month → year, month"""
    BROKEN_CHARS = "?\ufffd" 
    df.info()
    df.columns = df.columns.str.strip() # 컬럼 이름의 공백 제거
    df.columns = df.columns.str.lower() # 컬럼 이름의 공백 제거

    text_cols = df.select_dtypes(include=["object", "string"]).columns

    pattern = r"(?<=[가-힣])[{re.escape(BROKEN_CHARS)}](?=[가-힣])"

    #컬럼 내용 전처리
    for col in text_cols:
        df[col] = df[col].str.normalize("NFC")
        df[col] = df[col].str.replace(pattern, "", regex=True)  # "기타음?식료품" → "기타음식료품"
        df[col] = df[col].replace(r"[\ufeff\u200b-\u200d]", "", regex=True) 
        df[col] = df[col].replace(r"[\xa0\u3000]", " ", regex=True)
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

    
    cols = [str(col) for col in df.columns]
    date = set(cols) & set_date  #중첩된 for문으로 해결할 것
    if not date:
        print("표에 날짜 형식이 존재하지 않음") 
    else:
        year_month= date.pop() #수정
        df[year_month] = pd.to_datetime(df[year_month],format = "%Y%m",errors="coerce") #%Y-%m,%Y%m%d




    return df

prepare_data(df)
#print(df.shape)
#print(df.dtypes)
#print(df.head())
#print(df["biz_nm"].nunique())
#print(df["user_type"].unique())



