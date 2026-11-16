"""
tools.py는 에이전트가 사용하는 도구들의 모음입니다
여기서 명시하는 도구란 평범란 파이썬 함수이며 LLM은 이 함수를 직접 실행하지 못하고 "이 함수를 이 값으로 불러줘"라는 요청만 합니다
실질적인 실행은 agent.py가 맡습니다



도구 생성의 규칙
1.입력은 단순한 값(str,int,list)만 받습니다 ,출력은 dict입니다
2실패해도 예외를 던지지 말고 {error:"이유"}를 돌려주도록 합시다 
3원본 데이터가 아니라 scripts/prepare_data.py로 미리 정의해준 data/csv만 읽도록 설계합시다 

전체적인 흐름 
1.CSV -> preprocessing.read_any_csv -> clean_table -> register_table(메모리에 저장)
  질문 -> describe_data -> filter-and_aggregate(결과는 RESULTS에 보관 ,llm에는 일부만 )-> make_chart

"""

import difflib
import json
from collections import OrderedDict
from functools import lru_cache

from pathlib import Path

import pandas as pd
from google.genai import types

from preprocessing import clean_table,read_any_csv

DATA_DIR = Path(__file__).parent / "data" #데이터 경로 저장
MAX_ROWS = 30  #llm에 돌려줄 표의 최대 행수 (화면에는 전체를 보여준다 )
MAX_CATEGORY_VALUES = 60 #describe_data에서 버여줄 최대 범주 컬럼의 값
MAX_RESULTS = 50 #RESULTS에 보관할 집계 수
ALLOWED_AGGS = {"sum": "합계", "mean": "평균", "count": "행 수", "max": "최댓값", "min": "최솟값"}
UNKNOWN_LIKE = {"알수없음", "알 수 없음", "미상", "기타", "unknown"}  

# 앱에서 쓸 수 있는 표 이름 목록 (data/ 폴더의 파일 이름, 확장자 제외)
COLUMN_LABELS = {
    # 카드 이용 현황 (제주데이터허브)
    "base_year_month": ("기준 연월", ""),
    "biz_nm": ("업종명", ""),
    "user_type": ("이용자 유형", ""),
    "sex": ("성별", ""),
    "user_count": ("이용자 수", "명"),
    "use_count": ("이용 건수", "건"),
    "use_amount": ("이용 금액", "원"),
    # 전처리가 날짜에서 만드는 컬럼 (모든 데이터 공통)
    "year": ("연도", ""),
    "month": ("월", ""),
    "period": ("연월(YYYY-MM)", ""),
    "weekday": ("요일", ""),
}

BUILTIN_INFO = {
    #{" 짧은 제목(key)": {"title": "파일 제목", "source" : "제주데이터허브 「허브 페이지의 데이터셋 이름」 "}}
}

TABLES : dict[str,dict] = {} # 표 이름
RESULTS : "OrderedDict[str,pd.DataFrame]" = OrderedDict()
_result_counter = 0

def label(col:str) -> str:
    """컬럼 이름 -> 한글 뜻(단위), 사전에 없으면 영문 그대로 """

    if col not in COLUMN_LABELS:
        return col

    ko, unit = COLUMN_LABELS[col]
    return f"{ko}({unit})" if unit else ko  #유닛이 없으면 ko를 반환


def register_table(name: str , source, title :str |None = None , source_label : str | None = None) -> dict:
    """CSV를 또는 업로드 바이트를 읽고 공통 전처리를 거친다음에 TABLE에 등록한다 
    
    내장 데이터와 업로드 데이터가 모두 이 함수 하나를 거친다
    output:{"name","report"} 또는 {"error"}
    """

    try: 
        raw,encoding = read_any_csv(source)
        df, report = clean_table(raw)

    except Exception as e : #어떤 csv가 와도 앱이 멈추지 않게 
        return  {"error" : f"CSV를 읽지 못했습니다 : {e}"}
    if df.empty:
        return {"error" : "CSV에는 데이터 행이 없습니다 "}

    report["encoding"] = encoding
    TABLES[name] = {
        "df": df,
        "title": title or name,
        "source": source_label or f"이용자 제공 파일 ({name})",
        "report": report,
        "describe": None,  # describe_data 결과를 처음 한 번만 계산해 저장 (캐시)
    }
    return {"name": name, "report": report}   


def load_builtin_tables() -> list[dict]:
    """data 폴더의 존재하는 CSV를 전부 등록한다 , 앱 시작할때 한 번 부른다"""
    results = []
    for path in sorted(DATA_DIR.glob("*.csv")):  # 데이터 폴더 하위에 존재하는 csv를 한번씩 부른다
        info = BUILTIN_INFO.get(path.stem,{})
        results.append(register_table(
            name = path.stem,
            source=path,
            title=info.get("title"),
            source_label=f"{info.get('source', '제주데이터허브 공개 데이터')}, 파일 {path.name}",
        ))

    return results


def get_result(result_id : str) -> pd.DataFrame | None :
    """app.py가 접근하며 를 집계 결과 전체를꺼낼 때 사용한다 """
    return RESULTS.get(result_id)



def _to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON으로 보낼 수 있는 dict 목록을 반환
    데이터 프레임 -> JSON -> 순수 파이썬 타입"""

    return json.loads(df.to_json(orient = "records", force_ascii= False))

def _get_table(name :str): 
    """이름으로 표를 찾는다 ,없으면 (None, error dict)"""
    if name not in TABLES:
        return None,{"error": f"'{name}' 표가 없습니다. 사용할 수 있는 표: {list(TABLES)}"}
    return TABLES[name], None


#도구 1 데이터 설명
def describe_data(name: str) -> dict:
    """표의 컬럼(한글 뜻,종류,값 목록),행 수,기간을 알려준다 
    LLM은 처음엔 컬럼 이름도 정확한 값 이름도 모른다 
    그래서 LLM이 이 도구를 먼저 호출해서 어떤 컬럼으로 거르고 묶을 수 있는지 값은 뭐라고 적혀 있는지를 파악한다 """

    table, err = _get_table(name)
    if err:
        return err
    assert table is not None #나의 보증
    if table["describe"] is not None:
        return table["describe"]

    df = table["df"]
    date_info = table["report"]["date"]
    columns = []

    for col in df.columns:
        s = df[col] 
        item = {"column": col , "meaning": label(col)} #의미를 적어나 llm도 이해할 수 있게
        
        if date_info and col == date_info["column"]:  #들어온 숫자가 날짜면..
            item["kind"] = f"날짜 원본 ({date_info['unit']} 단위). 거르기·묶기는 year, month, period를 쓸 것"

        elif pd.api.types.is_numeric_dtype(s):  #들어온 컬럼이 숫자면 하는 처리
            item["kind"] = "숫자"
            item["min"] , item["max"] = s.min(),s.max()

            #년도,월처럼 범위가 적은 숫자,컬럼은 목록 보여주기
            if s.nunique() <= 31:
                item["values"] = sorted(s.dropna().unique().tolist())
        else:
            uniques = s.dropna().unique()
            item["kind"] = "범주"
            item["n_values"] = len(uniques)
            item["values"] = sorted(uniques.tolist())[:MAX_CATEGORY_VALUES]
            if len(uniques) > MAX_CATEGORY_VALUES:
                item["values_truncated"] = True
        columns.append(item)

    result = {
        "name": name,
        "title": table["title"],
        "rows": len(df),
        "columns": columns,
        "source": table["source"],
    }
    if date_info and "period" in df.columns:
        result["period"] = f"{df['period'].min()} ~ {df['period'].max()}"

    result = json.loads(json.dumps(result, default=int, ensure_ascii=False))  # numpy 숫자 → int
    table["describe"] = result
    return result


# ---------------------------------------------------------------------------
# 도구 2. 거르기 + 묶기 + 집계
# ---------------------------------------------------------------------------
def _build_mask(df: pd.DataFrame, filters: list[dict]):
    """filters = [{"column": "year", "values": ["2024"]}, ...] → 행 선택 마스크. 실패하면 (None, error).
    
    build_mask가 각 행이 조건에 맞는지 True/False로 표시한 목록(마스크)을 만들어 넘깁니다"""
    mask = pd.Series(True, index=df.index)
    for f in filters:
        col, values = f.get("column"), f.get("values")
        if col not in df.columns:
            return None, {"error": f"filters의 '{col}' 컬럼이 없습니다. 컬럼 목록: {list(df.columns)}"}
        if not isinstance(values, list) or not values:
            return None, {"error": f"filters의 '{col}' values는 값 목록이어야 합니다 (예: [\"2024\"])"}

        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            # [학습 포인트] LLM은 숫자도 "2024"처럼 글자로 보낼 때가 많다. 숫자 컬럼이면 숫자로 바꿔 비교.
            try:
                wanted = [float(v) for v in values]
            except (TypeError, ValueError):
                return None, {"error": f"'{col}'은 숫자 컬럼입니다. 숫자 값을 주세요: {values}"}
        else:
            wanted = [str(v).strip() for v in values]
            existing = set(s.dropna().unique())
            missing = [v for v in wanted if v not in existing]
            if missing:
                # 비슷한 값을 찾아 알려 주면 LLM이 '한식' → '한식 음식점업'처럼 고쳐서 다시 부른다
                hints = {v: difflib.get_close_matches(v, list(existing), n=3, cutoff=0.3) for v in missing}
                return None, {"error": f"'{col}'에 없는 값: {missing}. 비슷한 값: {hints}"}
        mask &= s.isin(wanted)
    return mask, None


def filter_and_aggregate(
    name: str,
    metric: str,
    agg: str = "sum",
    group_by: list[str] | None = None,
    filters: list[dict] | None = None,
    sort: str | None = None,
    top_n: int | None = None,
    share: bool = False,
) -> dict:
    """조건으로 거르고(filters) → 묶어서(group_by) → 집계(agg)한다.

    예) 2024년 이용 금액 상위 5개 업종
        filters=[{"column": "year", "values": ["2024"]}], group_by=["biz_nm"],
        metric="use_amount", agg="sum", sort="desc", top_n=5
    """

    global _result_counter

    table, err = _get_table(name)
    if err:
        return err
    assert table is not None
    df = table["df"]
    group_by = group_by or []
    filters = filters or []

    # 1) 입력 검사 → 틀리면 error (LLM이 고쳐서 다시 부르게)
    if agg not in ALLOWED_AGGS:
        return {"error": f"agg는 {list(ALLOWED_AGGS)} 중 하나여야 합니다"}
    bad = [c for c in group_by + [metric] if c not in df.columns]
    if bad:
        return {"error": f"없는 컬럼: {bad}. 컬럼 목록: {list(df.columns)}"}
    if agg != "count" and not pd.api.types.is_numeric_dtype(df[metric]):
        return {"error": f"'{metric}'은 숫자 컬럼이 아니라 {agg}를 할 수 없습니다 (행 수는 agg='count')"}
    if sort not in (None, "desc", "asc"):
        return {"error": "sort는 'desc'(큰 값부터) 또는 'asc'(작은 값부터)"}

    # 2) 거르기
    mask, err = _build_mask(df, filters)
    if err:
        return err
    sub = df[mask]
    if sub.empty:
        return {"rows": [], "row_count": 0, "note": "조건에 맞는 행이 없습니다", "source": table["source"]}

    # 3) 묶기 + 집계
    # [학습 포인트] groupby는 '같은 값끼리 모아서 → 각 묶음에 함수 적용'. 결과는 묶음 하나당 한 행.
    if group_by:
        res = sub.groupby(group_by, dropna=False)[metric].agg(agg).reset_index()
    else:
        res = pd.DataFrame({metric: [sub[metric].agg(agg)]})
    if agg == "mean":
        res[metric] = res[metric].astype(float).round(2)

    # 4) 비중(%): 상위 N개로 자르기 '전'에 전체 합 기준으로 계산
    if share:
        total = res[metric].sum()
        res["share_pct"] = (res[metric] / total * 100).round(1) if total else 0.0

    # 5) 정렬 + 상위 N개
    if sort:
        res = res.sort_values(metric, ascending=(sort == "asc"), kind="stable")
    if top_n:
        res = res.head(int(top_n))
    res = res.reset_index(drop=True)

    # 6) 결과 전체는 RESULTS에 보관 → 그래프·CSV 저장은 이 전체를 씀
    #    LLM에게는 MAX_ROWS행까지만 보냄 (토큰 절약)
    _result_counter += 1
    result_id = f"r{_result_counter}"
    RESULTS[result_id] = res
    while len(RESULTS) > MAX_RESULTS:
        RESULTS.popitem(last=False)  # 가장 오래된 결과부터 지움

    # 답변의 '기준'으로 쓸 설명 문장 (LLM이 지어내지 않도록 코드가 만들어 줌)
    cond = ", ".join(f"{label(f['column'])}={'/'.join(map(str, f['values']))}" for f in filters) or "전체"
    basis = (f"조건: {cond} / 묶음: {', '.join(label(c) for c in group_by) or '없음'} / "
             f"값: {label(metric)} {ALLOWED_AGGS[agg]} / 사용한 원본 행 {len(sub):,}개")

    notes = []
    for c in group_by:
        found = UNKNOWN_LIKE & set(map(str, res[c].unique()))
        if found:
            notes.append(f"'{label(c)}'에 {sorted(found)} 항목이 포함됨 (값이 수집되지 않았거나 분류되지 않은 건)")
    if top_n and len(res) == int(top_n):
        notes.append(f"상위 {top_n}개만 표시")

    return {
        "result_id": result_id,
        "basis": basis,
        "row_count": len(res),
        "rows": _to_records(res.head(MAX_ROWS)),
        "truncated": len(res) > MAX_ROWS,
        "notes": notes,
        "source": table["source"],
    }


# ---------------------------------------------------------------------------
# 도구 3. 그래프 요청
# ---------------------------------------------------------------------------
def make_chart(result_id: str, kind: str, x: str, y: str, title: str) -> dict:
    """filter_and_aggregate 결과(result_id)로 그래프를 그려 달라는 '요청서'를 만든다.

    [학습 포인트] 그래프 데이터를 LLM이 다시 적어 보내게 하지 않고 result_id로 가리키게 한 이유
      1) LLM이 숫자를 옮기다 틀릴 수 있다 (신뢰성)  2) 90개월치 숫자를 다시 보내면 토큰 낭비
    그림은 화면(app.py)이 RESULTS[result_id] 전체로 그린다.
    """
    df = get_result(result_id)
    if df is None:
        return {"error": f"'{result_id}' 결과가 없습니다. filter_and_aggregate를 먼저 부르세요"}
    if kind not in ("bar", "line"):
        return {"error": "kind는 'bar'(항목 비교) 또는 'line'(시간에 따른 변화)"}
    bad = [c for c in (x, y) if c not in df.columns]
    if bad:
        return {"error": f"결과에 없는 컬럼: {bad}. 결과 컬럼: {list(df.columns)}"}
    return {
        "chart": {"result_id": result_id, "kind": kind, "x": x, "y": y, "title": title,
                  "x_label": label(x), "y_label": label(y)},
        "message": "그래프를 화면에 표시했습니다",
    }


# ---------------------------------------------------------------------------
# LLM에게 보여 줄 '도구 설명서'
# LLM은 함수 코드를 보지 못하고, 아래 이름·설명·입력 형식만 보고 언제 무엇을 부를지 정한다.
# [학습 포인트] 표 목록은 CSV 업로드로 바뀌므로, 고정 상수가 아니라 '부를 때마다 만드는 함수'로 둔다.
# ---------------------------------------------------------------------------
def get_tool_declarations() -> list[types.FunctionDeclaration]:
    table_names = list(TABLES) or ["(등록된 표 없음)"]
    name_param = {"type": "string", "enum": table_names, "description": "표 이름"}

    return [
        types.FunctionDeclaration(
            name="describe_data",
            description="표의 컬럼 이름·한글 뜻·종류, 범주 컬럼의 정확한 값 목록, 행 수, 기간을 확인한다. "
                        "컬럼 이름이나 값 이름을 모르면 가장 먼저 호출한다.",
            parameters_json_schema={
                "type": "object",
                "properties": {"name": name_param},
                "required": ["name"],
            },
        ),
        types.FunctionDeclaration(
            name="filter_and_aggregate",
            description="조건으로 행을 거르고, 컬럼으로 묶어서, 숫자 컬럼을 집계한다. 답에 쓰는 모든 숫자는 이 도구 결과에서 가져온다. "
                        "결과의 result_id는 make_chart에 쓴다.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "name": name_param,
                    "metric": {"type": "string", "description": "집계할 컬럼 (예: 이용 금액 컬럼)"},
                    "agg": {"type": "string", "enum": list(ALLOWED_AGGS),
                            "description": "sum 합계, mean 평균, count 행 수, max 최댓값, min 최솟값"},
                    "group_by": {"type": "array", "items": {"type": "string"},
                                 "description": "묶을 컬럼 목록. 연도별은 year, 월별(연도 합산)은 month, 월별 추이는 period"},
                    "filters": {
                        "type": "array",
                        "description": "거르기 조건 목록. 값은 describe_data의 values에 있는 그대로 적는다",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "values": {"type": "array", "items": {"type": "string"},
                                           "description": "이 값들 중 하나인 행만 남김"},
                            },
                            "required": ["column", "values"],
                        },
                    },
                    "sort": {"type": "string", "enum": ["desc", "asc"],
                             "description": "순위 질문이면 desc. 시간 순서 결과에는 쓰지 않는다"},
                    "top_n": {"type": "integer", "description": "상위 몇 개만 (sort와 함께)"},
                    "share": {"type": "boolean", "description": "true면 전체 대비 비중(%) 컬럼 share_pct 추가"},
                },
                "required": ["name", "metric", "agg"],
            },
        ),
        types.FunctionDeclaration(
            name="make_chart",
            description="filter_and_aggregate 결과를 그래프로 화면에 표시한다. 추이는 line, 항목 비교는 bar.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "result_id": {"type": "string", "description": "filter_and_aggregate가 돌려준 result_id"},
                    "kind": {"type": "string", "enum": ["bar", "line"]},
                    "x": {"type": "string", "description": "x축 컬럼 (결과의 컬럼 이름)"},
                    "y": {"type": "string", "description": "y축 컬럼 (결과의 컬럼 이름)"},
                    "title": {"type": "string", "description": "그래프 제목 (한국어)"},
                },
                "required": ["result_id", "kind", "x", "y", "title"],
            },
        ),
    ]


# LLM이 돌려준 '도구 이름'으로 실제 함수를 찾기 위한 표
TOOL_FUNCTIONS = {
    "describe_data": describe_data,
    "filter_and_aggregate": filter_and_aggregate,
    "make_chart": make_chart,
}


    