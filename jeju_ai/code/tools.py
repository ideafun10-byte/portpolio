"""
tools.py — 표 등록 + 에이전트가 쓰는 '도구' 3개 + LLM에게 보여 줄 도구 설명서

도구 = 평범한 파이썬 함수입니다.
LLM은 이 함수를 직접 실행하지 못하고, "이 함수를 이 값으로 불러 줘"라고 '요청'만 합니다.
실제 실행은 agent.py가 맡습니다.

도구 규칙
- 입력은 단순한 값(str, int, bool, list)만. LLM이 JSON으로 값을 넘겨주기 때문.
- 출력은 dict. 이 dict가 그대로 LLM에게 돌아가므로 '작게' 만든다(행 수 제한 → 토큰 절약).
- 실패해도 예외를 던지지 않고 {"error": "이유"}를 돌려준다 → LLM이 읽고 고쳐서 다시 부를 수 있음.
- 툴 코드에 특정 데이터의 컬럼 이름을 쓰지 않는다 (어떤 CSV든 같은 툴로).
  컬럼의 '한글 뜻'은 COLUMN_LABELS(데이터 사전)에만 적는다.

전체 흐름
  CSV → preprocessing.read_any_csv → clean_table → register_table (메모리에 보관)
  질문 → describe_data → filter_and_aggregate (결과는 RESULTS에 보관, LLM에겐 일부만) → make_chart
"""
import difflib
import json
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from google.genai import types

from preprocessing import clean_table, read_any_csv

DATA_DIR = Path(__file__).parent.parent / "data"
# [수정 3-나] 내장 데이터를 찾는 곳: data/ 바로 아래 + scripts/convert_raw.py가 만든 data/<이름>_utf8/ 폴더
#   원본 폴더(data/raw, data/hub 등)는 읽지 않는다 → 같은 데이터가 두 번 등록되는 것을 막음
CONVERTED_SUFFIX = "_utf8"
EXCLUDE_FILES = {"data_dictionary.csv"}  # 분석할 표가 아니라 '컬럼 사전' 파일이므로 표로 등록하지 않음
MAX_ROWS = 30  # LLM에게 돌려줄 표의 최대 행 수 (화면에는 전체를 보여 줌)
MAX_CATEGORY_VALUES = 60  # describe_data에서 범주 컬럼의 값 목록을 몇 개까지 보여 줄지
MAX_RESULTS = 50  # RESULTS에 보관할 집계 결과 수 (오래된 것부터 지움 → 메모리 절약)
MAX_NAME_LEN = 60
ALLOWED_AGGS = {"sum": "합계", "mean": "평균", "count": "행 수", "max": "최댓값", "min": "최솟값"}
UNKNOWN_LIKE = {"알수없음", "알 수 없음", "미상", "기타", "unknown"}  # 결과에 있으면 한계 문구를 붙일 값

# ---------------------------------------------------------------------------
# 데이터 사전: 영문 컬럼 → (한글 뜻, 단위)
# 내부 컬럼명은 영문 그대로 두고, 화면·답변에서만 한글을 쓴다.
# TODO(개선): 명세서 기반 사전(build_dictionary.py)을 만들면 여기에 합치기
# ---------------------------------------------------------------------------
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

# 내장 데이터의 화면용 이름과 출처. 키 = data 폴더의 파일 이름(확장자 제외)
# TODO: 실제 파일 이름으로 바꾸기. 여기에 없는 파일도 등록은 되고, 제목·출처는 파일 이름으로 표시됨
BUILTIN_INFO = {
    "card_usage": {
        "title": "제주 업종별·이용자 유형별·성별 카드 이용 현황 (월간)",
        "source": "제주데이터허브 공개 데이터",
    },
}

# ---------------------------------------------------------------------------
# 표 보관소
# [학습 포인트] 모듈 수준 dict는 프로그램이 켜져 있는 동안 계속 유지된다.
#   CSV를 질문마다 다시 읽지 않고 한 번 정리해서 메모리에 두는 이유: 디스크 읽기·전처리는 느리고,
#   질문마다 반복하면 시간 낭비. (28,000행 정도는 메모리 몇 MB 수준이라 부담 없음)
# ---------------------------------------------------------------------------
TABLES: dict[str, dict] = {}  # 표 이름 → {"name", "df", "title", "source", "report", "describe"}
RESULTS: "OrderedDict[str, pd.DataFrame]" = OrderedDict()  # 결과 id → 집계 결과 전체
_result_counter = 0
# [수정 3-가] '내장 데이터를 이미 읽었는가'를 TABLES와 같은 파일(모듈)에 둔다.
#   None = 아직 한 번도 안 읽음.  dict = 읽은 결과 (파일 경로 → 등록 결과)
_builtin_results: dict[str, dict] | None = None


def label(col: str) -> str:
    """컬럼 이름 → '한글 뜻(단위)'. 사전에 없으면 영문 그대로."""
    if col not in COLUMN_LABELS:
        return col
    ko, unit = COLUMN_LABELS[col]
    return f"{ko}({unit})" if unit else ko


# ---------------------------------------------------------------------------
# [수정 3-다] 표 이름 만들기 / 찾기
# ---------------------------------------------------------------------------
def safe_table_name(text: str) -> str:
    """파일 이름(확장자 제외) → 표 이름. 한글·영문·숫자·_ 만 남긴다.

    [학습 포인트] LLM은 도구를 부를 때 표 이름을 '글자 그대로' 적어야 한다.
      '제주시 유동인구(2024).v2' 처럼 공백·괄호·점이 섞인 이름은 LLM이 한 글자만 다르게 적어도 못 찾는다.
      그래서 등록할 때 미리 단순한 이름으로 바꿔 둔다. (내장 파일과 업로드 파일이 같은 규칙을 씀)
    """
    text = unicodedata.normalize("NFC", str(text))  # Mac에서 온 파일 이름의 풀어진 한글(NFD)을 합침
    name = re.sub(r"[^0-9A-Za-z가-힣_]+", "_", text).strip("_")[:MAX_NAME_LEN]
    return name or "table"


def _norm(text) -> str:
    """이름 '비교'용: 끝의 .csv를 떼고, 공백·기호를 빼고, 소문자로 맞춘다."""
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    text = re.sub(r"\.csv$", "", text)
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def resolve_table_name(name) -> str | None:
    """LLM이 적어 보낸 이름으로 실제 표 이름을 찾는다. 못 찾으면 None.

    찾는 순서: ① 글자 그대로 같은 이름 → ② 공백·기호·대소문자만 다른 이름 또는 '제목'
    [학습 포인트] '비슷한 이름'으로 추측해서 고르지는 않는다. 엉뚱한 표로 계산하면
      오류보다 더 나쁜 '틀린 숫자'가 나오기 때문. 애매하면 None → 도구가 error로 표 목록을 알려 준다.
    """
    if isinstance(name, str) and name in TABLES:
        return name
    key = _norm(name)
    if not key:
        return None
    matches = [n for n, t in TABLES.items() if key in (_norm(n), _norm(t["title"]))]
    return matches[0] if len(matches) == 1 else None


def register_table(name: str, source, title: str | None = None, source_label: str | None = None) -> dict:
    """CSV(경로 또는 업로드 bytes)를 읽고 공통 전처리를 거쳐 TABLES에 등록한다.

    내장 데이터와 업로드 데이터가 모두 이 함수 하나를 지나간다 → '똑같이 처리'가 보장됨.
    돌려주는 값: {"name", "report"} 또는 {"error"}
    """
    try:
        raw, encoding = read_any_csv(source)
        df, report = clean_table(raw)
    except Exception as e:  # 어떤 CSV가 와도 앱이 멈추지 않게
        return {"error": f"CSV를 읽지 못했습니다: {e}"}
    if df.empty:
        return {"error": "CSV에 데이터 행이 없습니다"}

    report["encoding"] = encoding
    TABLES[name] = {
        "name": name,
        "df": df,
        "title": title or name,
        "source": source_label or f"이용자 제공 파일 ({name})",
        "report": report,
        "describe": None,  # describe_data 결과를 처음 한 번만 계산해 저장 (캐시)
    }
    return {"name": name, "report": report}


def _builtin_files() -> list[Path]:
    """내장 데이터 후보 파일 목록. data/*.csv 먼저, 그다음 data/*_utf8/*.csv (폴더·파일 이름 순)."""
    if not DATA_DIR.exists():
        return []
    folders = [DATA_DIR] + sorted(p for p in DATA_DIR.iterdir()
                                  if p.is_dir() and p.name.endswith(CONVERTED_SUFFIX))
    files = []
    for folder in folders:
        files += sorted(
            p for p in folder.iterdir()
            # suffix.lower(): 'DATA.CSV' 처럼 확장자가 대문자인 파일도 읽음
            # '.'·'~$'로 시작하는 파일: 숨김 파일, 엑셀이 열려 있을 때 생기는 임시 파일 → 건너뜀
            if p.is_file() and p.suffix.lower() == ".csv"
            and p.name not in EXCLUDE_FILES and not p.name.startswith((".", "~$"))
        )
    return files

def _unique_name(base: str) -> str:
    """base가 이미 쓰이고 있으면 뒤에 _2, _3 … 을 붙여 겹치지 않는 표 이름을 만든다."""
    if base not in TABLES:  # 겹치지 않으면 그대로 사용
        return base
    i = 2
    # MAX_NAME_LEN - 3 = 글자 수 제한 안에 '_2' ~ '_99' 꼬리가 들어갈 자리를 남긴 길이
    while f"{base[:MAX_NAME_LEN - 3]}_{i}" in TABLES:  # _2도 쓰이고 있으면 _3, _4 … 로
        i += 1
    return f"{base[:MAX_NAME_LEN - 3]}_{i}"

def load_builtin_tables() -> list[dict]:
    """내장 CSV를 전부 등록한다. 이미 읽은 파일은 건너뛴다 → 여러 번 불러도 안전.

    돌려주는 값: 이번에 '새로' 시도한 파일마다
      {"file", "name", "report"} 성공 / {"file", "error"} 실패 / {"file", "skipped"} 일부러 건너뜀

    [수정 11/16-2] 서로 다른 파일이 같은 표 이름으로 바뀌는 경우
      ① 파일 이름이 다르면 _2를 붙여 둘 다 등록
      ② 파일 이름까지 같으면(다른 폴더의 같은 파일) 건너뛰되 그 사실을 결과에 남긴다
    """
    results = []
    loaded_files = {t.get("file") for t in TABLES.values()}  # 이미 읽은 파일 경로
    stems = {t["stem"]: t["file"] for t in TABLES.values() if t.get("stem")}  # 파일 이름(확장자 제외) → 경로
    for path in _builtin_files():
        rel = str(path.relative_to(DATA_DIR))  # data 폴더 기준 경로 (예: hub_utf8\card_usage.csv)
        if rel in loaded_files:  # 같은 파일을 다시 읽으려는 경우 (rescan=True) → 조용히 넘어감
            continue
        if path.stem in stems:  # 다른 폴더에 이름이 똑같은 파일 → 한 번만 등록하고 기록을 남김
            results.append({"file": rel, "skipped": f"이름이 같은 파일이 이미 등록되어 건너뜀 ({stems[path.stem]})"})
            continue
        name = _unique_name(safe_table_name(path.stem))  # 표 이름이 겹치면 _2, _3을 붙임
        info = BUILTIN_INFO.get(path.stem, {})
        res = register_table(
            name=name,
            source=path,
            title=info.get("title") or path.stem,  # 사전에 없으면 파일 이름을 제목으로
            source_label=f"{info.get('source', '제주데이터허브 공개 데이터')}, 파일 {path.name}",
        )
        res["file"] = rel
        if "error" in res:  # 어느 파일에서 난 오류인지 알 수 있게 파일 이름을 앞에 붙임
            res["error"] = f"{rel}: {res['error']}"
        else:  # 어느 파일에서 온 표인지 표 보관소에도 적어 둠 (다음 파일과 겹치는지 확인할 때 씀)
            TABLES[name].update(file=rel, stem=path.stem)
            loaded_files.add(rel)
            stems[path.stem] = rel
        results.append(res)
    return results


def ensure_builtin_tables(rescan: bool = False) -> list[dict]:
    """내장 데이터가 아직 안 올라와 있으면 읽고, 이미 읽었으면 아무 일도 하지 않는다.

    화면(app.py)이 다시 실행될 때마다 불러도 된다. rescan=True면 data 폴더를 다시 훑어 새 파일만 추가한다.

    [학습 포인트] 예전에는 app.py에서 @st.cache_resource로 '앱이 켜져 있는 동안 딱 한 번'만 읽었다.
      그런데 Streamlit은 .py 파일이 저장되면 우리 모듈(tools, agent 등)을 메모리에서 지우고 다시 import한다.
      → TABLES는 빈 dict로 새로 만들어지는데, 캐시는 '이미 실행했음'이라고 기억해서 다시 읽지 않는다.
      → 화면에는 데이터가 없다고 나온다.
      '읽었는지 표시(_builtin_results)'를 TABLES와 같은 모듈에 두면, 모듈이 새로 만들어질 때
      표시도 함께 None으로 돌아가므로 자동으로 다시 읽는다. (둘의 수명이 같아짐)
    """
    global _builtin_results
    with _load_lock:  # 이 안은 한 번에 한 접속만 실행 (먼저 들어간 쪽이 끝나야 다음이 들어감)
        if _builtin_results is None or rescan:  # 검사도 잠금 '안'에서 해야 함
            found = dict(_builtin_results or {})
            for res in load_builtin_tables():
                found[res["file"]] = res  # 같은 파일을 다시 시도했으면 새 결과로 바꿈
            _builtin_results = found
        return list(_builtin_results.values())


def get_result(result_id: str) -> pd.DataFrame | None:
    """화면(app.py)이 집계 결과 '전체'를 꺼낼 때 쓴다."""
    return RESULTS.get(result_id)


def _to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → JSON으로 보낼 수 있는 dict 목록.

    [학습 포인트] numpy 정수, pandas의 <NA> 같은 값은 그대로 LLM API에 보내면 직렬화 오류가 난다.
      to_json → json.loads 를 한 번 거치면 전부 파이썬 기본 타입(int, float, str, None)이 된다.
    """
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _get_table(name: str):
    """이름으로 표를 찾는다. 없으면 (None, error dict)."""
    real = resolve_table_name(name)
    if real is None:
        # 이름과 제목을 함께 알려 줘야 LLM이 '제목'이 아니라 '이름'을 골라 다시 부를 수 있다
        listing = [f"{n} ({t['title']})" for n, t in TABLES.items()]
        return None, {"error": f"'{name}' 표가 없습니다. 표 이름은 다음 중 하나를 글자 그대로 쓰세요 "
                               f"(괄호 안은 제목): {listing}"}
    return TABLES[real], None


# ---------------------------------------------------------------------------
# 도구 1. 데이터 설명
# ---------------------------------------------------------------------------
def describe_data(name: str) -> dict:
    """표의 컬럼(한글 뜻·종류·값 목록), 행 수, 기간을 알려 준다.

    LLM은 처음엔 컬럼 이름도, '한식 음식점업' 같은 정확한 값 이름도 모른다.
    그래서 이 도구를 먼저 불러 '어떤 컬럼으로 거르고 묶을 수 있는지, 값은 정확히 뭐라고 적혀 있는지'를 본다.
    """
    table, err = _get_table(name)
    if err:
        return err
    assert table is not None
    if table["describe"] is not None:  # 이미 계산했으면 재사용
        return table["describe"]

    df = table["df"]
    date_info = table["report"]["date"]
    columns = []
    for col in df.columns:
        s = df[col]
        item = {"column": col, "meaning": label(col)}
        if date_info and col == date_info["column"]:
            item["kind"] = f"날짜 원본 ({date_info['unit']} 단위). 거르기·묶기는 year, month, period를 쓸 것"
        elif pd.api.types.is_numeric_dtype(s):
            item["kind"] = "숫자"
            item["min"], item["max"] = s.min(), s.max()
            # 연도·월처럼 값 종류가 적은 숫자 컬럼은 값 목록도 보여 줌 (거르기에 쓰이므로)
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
        "name": table["name"],  # LLM이 살짝 다르게 적어 보냈어도 '진짜 이름'을 돌려줌
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
    """filters = [{"column": "year", "values": ["2024"]}, ...] → 행 선택 마스크. 실패하면 (None, error)."""
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
    name_param = {"type": "string", "enum": table_names,
                  "description": "표 이름. 표 목록에 적힌 이름을 글자 그대로 (제목이 아님)"}

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