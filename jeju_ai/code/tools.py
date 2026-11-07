"""
tools.py는 에이전트가 사용하는 도구들의 모음입니다
여기서 명시하는 도구란 평범란 파이썬 함수이며 LLM은 이 함수를 직접 실행하지 못하고 "이 함수를 이 값으로 불러줘"라는 요청만 합니다
실질적인 실행은 agent.py가 맡습니다



도구 생성의 규칙
1.입력은 단순한 값(str,int,list)만 받습니다 ,출력은 dict입니다
2실패해도 예외를 던지지 말고 {error:"이유"}를 돌려주도록 합시다 
3원본 데이터가 아니라 scripts/prepare_data.py로 미리 정의해준 data/csv만 읽도록 설계합시다 
"""
from functools import lru_cache

from pathlib import Path

import pandas as pd
from google.genai import types

DATA_DIR = Path(__file__).parent / "data" #데이터 경로 저장
MAX_ROWS = 20

# 앱에서 쓸 수 있는 표 이름 목록 (data/ 폴더의 파일 이름, 확장자 제외)
# TODO: 11/7 전처리가 끝나면 실제 이름으로 채우기 (예: "bus_usage_monthly")
AVAILABLE_TABLES: list[str] = []

@lru_cache(maxsize=None)
def load_table(name: str) -> pd.DataFrame:
    """data/<sub>/<name>.csv를 읽어 DataFrame으로 돌려준다"""

    # TODO: 파일 경로 만들기 → pd.read_csv
    #       한글 CSV는 인코딩이 'utf-8-sig' 또는 'cp949'일 수 있음
    raise NotImplementedError


#도구 1:데이터 설명
"""데이터가 어떻게 생겼는지 명시:컬럼 이름 ,타입,행 수,기간,예시 몇줄

llm은 처음엔 컬럼 이름을 모름,도구 1을 먼저 불러 데이터를 어떻게 거르고 묶을 지 확인 한 후 도구 2를 부룹니다 
"""

def describe_data(name: str) -> dict:
    raise NotImplementedError


#도구 2:거르기,묶기,집계

def filter_and_aggregate(
        name: str,
        group_by:list[str],
        metric: str,
        agg: str = "sum",
        filters: dict | None = None,
) -> dict:
    """조건으로 거르고(filter) → 묶어서(group by) → 합계/평균 등을 낸다.
 
    예) name="버스이용", filters={"읍면동": "애월읍"}, group_by=["년월"], metric="승차인원", agg="sum"
    """
    # TODO 1: 없는 컬럼·허용하지 않는 agg면 {"error": ...}  (LLM이 고쳐서 다시 부르게)
    # TODO 2: filters 적용
    # TODO 3: groupby + agg
    # TODO 4: 결과가 너무 크면 MAX_ROWS행까지만 + "잘렸음" 표시
    # TODO 5: {"rows": [...], "row_count": ..., "source": "데이터 출처"} 형태로 반환
    raise NotImplementedError



#도구 3.그래프 요청
def make_chart(title: str, kind: str , x : str, y : str , rows:list[dict] ) -> dict:
    """'그래프를 그려달라는 요청서'를 보낸다 
    LLM은 그래프를 글지 못하니 그림을 그리지 않고 화면(app.py)가 그릴 수 있도록
    {"chart":{....}}의 형태로 '설계도'만 넘겨줍니다 
    """

    # TODO: kind가 "line"/"bar" 중 하나인지, x·y가 rows에 있는 키인지 확인
    raise NotImplementedError




# ---------------------------------------------------------------------------
# LLM에게 보여 줄 '도구 설명서'
# LLM은 함수 코드를 보지 못하고, 아래 이름·설명·입력 형식만 보고 언제 무엇을 부를지 정합니다.
# 그래서 description을 얼마나 분명하게 쓰느냐가 에이전트 품질을 크게 좌우합니다.
# ---------------------------------------------------------------------------
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="describe_data",
        description="분석 전에 표의 컬럼, 기간, 예시 행을 확인한다. 컬럼 이름을 모를 때 가장 먼저 호출한다.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": AVAILABLE_TABLES,  # 고를 수 있는 값을 제한하면 엉뚱한 이름을 덜 지어냄
                    "description": "표 이름",
                },
            },
            "required": ["name"],
        },
    ),
    # TODO: filter_and_aggregate 설명서 (group_by는 배열, filters는 객체)
    # TODO: make_chart 설명서
]
 
# LLM이 돌려준 '도구 이름'으로 실제 함수를 찾기 위한 표
TOOL_FUNCTIONS = {
    "describe_data": describe_data,
    "filter_and_aggregate": filter_and_aggregate,
    "make_chart": make_chart,
}