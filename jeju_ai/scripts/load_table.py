"""표 이름을 받아 정리된 DATAFrame을 반환하는 함수이다
표 이름 → 파일 경로 만들기 (data/card_usage.csv)
CSV를 dtype=str로 읽기
방금 만든 clean_table 실행 (글자 정리 + year, month)
정리된 DataFrame 돌려주기
"""
from pathlib import Path
import pandas as pd
import functools
import prepare_data

# 표 이름(영문) → 원래 파일 이름(한글)
# 파일 이름을 표 이름으로 바꿔 data/tables/에 넣었다면 오른쪽은 기록용(출처 표시, catalog)


#(LLM이 인식하는 이름:실제 이름)
TABLE_FILES = {
    "edu_private_cost":      "시도별학생1인당월평균사교육비정보.csv",
    "card_usage":            "업종별성별카드이용정보.csv",
    "bus_hourly_boarding":   "일별시간대별버스승차인원수현황.csv",
    "public_offices":        "제주관내관공서및주요기관현황.csv",
    "weather":               "제주기상개황정보.csv",
    "domestic_floating_pop": "제주도방문내국인유동인구추이.csv",
    "pm10_monthly":          "제주월별미세먼지(PM10)대기오염도정보.csv",
    "emergency_hospitals":   "제주응급실보유병원정보.csv",
    "high_schools":          "제주일반고등학교(국·공립)현황.csv",
    "seogwipo_clean_house":  "제주특별자치도 서귀포시_클린하우스현황_20240823.csv",
    "tourist_spot_visitors": "제주특별자치도_내국인관광지입장객수추이.csv",
    "restaurant_open_close": "제주특별자치도음식점목록(통합)_인허가일에따른영업및폐업음식점수.csv",
    "restaurants":           "제주특별자치도음식점목록(통합).csv",
    "beaches":               "제주특별자치도해수욕장정보.csv",
}

AVAILABLE_TABLES = list(TABLE_FILES)   # 표 이름 목록 (툴 설명서의 enum에 사용)


ROOT = Path(__file__).resolve().parent.parent   # scripts/ 의 한 칸 위 = jeju_ai/
DATA_DIR = ROOT / "data" / "raw"
OUT_SUFFIX = "_utf8"  # 결과 폴더 이름 끝에 붙는 글자 (원본 폴더와 구분하는 표시)

def load_table(table_name : str) ->pd.DataFrame:


    if table_name in AVAILABLE_TABLES:
        
    else:
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다 :{table_name}")
