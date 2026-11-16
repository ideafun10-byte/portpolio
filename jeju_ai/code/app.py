"""
app.py — 화면 담당 (계산은 tools.py, 판단은 agent.py에 맡김)

실행:  streamlit run code/app.py

화면 구성
  왼쪽(사이드바): ① API 키·모델 이름  ② 데이터 목록 + CSV 올리기
  가운데: 데이터 소개(LLM 안 씀) → 예시 질문 버튼 → 대화(답·그래프·표·계산 과정)

[학습 포인트] Streamlit은 사용자가 무언가 누르거나 입력할 때마다 이 파일 전체를 '위에서 아래로 다시' 실행한다.
  그래서
  - 지난 대화는 st.session_state(브라우저 탭을 닫기 전까지 유지되는 저장소)에 넣어 두고 매번 다시 그린다.
  - 한 번만 해야 하는 일(내장 CSV 읽기)은 @st.cache_resource로 감싸 처음 한 번만 실행한다.
"""
import re
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

from agent import DEFAULT_MODEL, run_agent
from tools import TABLES, describe_data, get_result, label, load_builtin_tables, register_table

st.set_page_config(page_title="제주 데이터 분석 도우미", layout="wide")


# ---------------------------------------------------------------------------
# 시작할 때 한 번: 내장 데이터(data/*.csv) 등록
# ---------------------------------------------------------------------------
@st.cache_resource  # 프로그램이 켜져 있는 동안 한 번만 실행 (CSV 읽기·전처리는 느리니까)
def init_builtin_tables() -> list[dict]:
    return load_builtin_tables()


builtin_results = init_builtin_tables()

if "messages" not in st.session_state:  #지금까지의 질문과 답 목록
    st.session_state.messages = []  # [{"role": "user", "content"} 또는 {"role": "assistant", **run_agent 결과}]

if "uploaded" not in st.session_state:  #이미 등록한 업로드 파일 기록
    st.session_state.uploaded = {}  # (파일 이름, 크기) → 등록한 표 이름. 같은 파일을 매번 다시 등록하지 않으려고

if "focus_table" not in st.session_state: #"데이터 소개"에 보여 줄 표 이름
    st.session_state.focus_table = next(iter(TABLES), None)  # '데이터 소개'에 보여 줄 표


# ---------------------------------------------------------------------------
# 작은 도우미 함수들
# ---------------------------------------------------------------------------
def plain(meaning: str) -> str:
    """'이용 금액(원)' → '이용 금액'. 예시 질문 문장에 단위 괄호가 끼면 어색해서 뺀다."""
    return re.sub(r"\(.*?\)", "", meaning).strip()


def table_name_from_file(file_name: str) -> str:
    """업로드 파일 이름 → 표 이름. LLM이 이름을 그대로 적어야 하므로 글자·숫자·_만 남긴다."""
    name = re.sub(r"[^0-9A-Za-z가-힣_]", "_", Path(file_name).stem)[:40] or "uploaded"
    return f"{name}_upload" if name in TABLES and name not in st.session_state.uploaded.values() else name


def example_questions(name: str) -> list[str]:
    """표의 컬럼 구성만 보고 예시 질문을 만든다. (LLM을 안 쓰므로 비용 0, 어떤 CSV에도 동작)"""
    d = describe_data(name)
    if "error" in d:
        return []
    cols = d["columns"]
    nums = [c for c in cols if c["kind"] == "숫자" and c["column"] not in ("year", "month")]
    cats = [c for c in cols if c["kind"] == "범주" and c["column"] not in ("period", "weekday")
            and 2 <= c.get("n_values", 0) <= 100]
    has_period = any(c["column"] == "period" for c in cols)
    if not nums:
        return []
    metric = plain(nums[-1]["meaning"])  # 보통 마지막 숫자 컬럼이 금액·건수 같은 핵심 값
    qs = []
    if cats:
        qs.append(f"{plain(cats[0]['meaning'])}별 {metric} 합계 상위 5개를 보여 줘")
    if has_period:
        qs.append(f"{metric}의 월별 추이를 그래프로 보여 줘")
    if len(cats) > 1:
        qs.append(f"{plain(cats[1]['meaning'])}별 {metric} 비중을 알려 줘")
    return qs


def draw_chart(spec: dict) -> None:
    """agent가 모은 chart 요청서로 그래프를 그린다. 데이터는 RESULTS의 '전체' 결과를 씀."""
    df = get_result(spec["result_id"])
    if df is None:
        st.caption("(오래된 결과라 그래프를 다시 그릴 수 없습니다)")
        return
    x, y = spec["x"], spec["y"]
    data = df[[x, y]].copy()
    data[x] = data[x].astype(str)  # 연도·월 숫자도 '이름표'로 다루기 (축에 2,024 처럼 나오지 않게)
    data[y] = pd.to_numeric(data[y], errors="coerce").astype(float)  # <NA> 같은 값은 그래프가 못 읽음

    # [학습 포인트] sort=None: 표에 있는 순서 그대로 그리기.
    #   기본값은 가나다순이라 '상위 5개' 순위나 월 순서가 뒤섞인다.
    if spec["kind"] == "line":
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=alt.X(f"{x}:O", sort=None, title=spec["x_label"]),
            y=alt.Y(f"{y}:Q", title=spec["y_label"]),
            tooltip=[alt.Tooltip(f"{x}:O", title=spec["x_label"]),
                     alt.Tooltip(f"{y}:Q", title=spec["y_label"], format=",.0f")],
        )
    else:  # 가로 막대: 업종 이름처럼 긴 한글 이름이 잘 보이게
        chart = alt.Chart(data).mark_bar().encode(
            y=alt.Y(f"{x}:N", sort=None, title=spec["x_label"]),
            x=alt.X(f"{y}:Q", title=spec["y_label"]),
            tooltip=[alt.Tooltip(f"{x}:N", title=spec["x_label"]),
                     alt.Tooltip(f"{y}:Q", title=spec["y_label"], format=",.0f")],
        )
    st.altair_chart(chart.properties(title=spec["title"]))


def render_assistant(msg: dict, idx: int) -> None:
    """에이전트 답 하나를 그린다. 새 답과 지난 답 모두 이 함수로 그려서 모양이 같게 유지된다."""
    if msg.get("error"):
        st.warning(msg["answer"])
        return
    st.markdown(msg["answer"])
    if msg["sources"]:
        st.caption("출처: " + " / ".join(msg["sources"]))  # 출처는 코드가 붙임 (LLM이 빠뜨려도 안전)

    for d in msg["displays"]:
        if d["type"] == "chart":
            draw_chart(d)

    tables = [d for d in msg["displays"] if d["type"] == "table"]
    if tables:
        with st.expander("표 보기 · CSV 저장"):
            for d in tables:
                df = get_result(d["result_id"])
                if df is None:
                    continue
                st.caption(d["basis"])
                shown = df.rename(columns={c: label(c) for c in df.columns})  # 화면에서만 한글 컬럼명
                st.dataframe(shown, hide_index=True)
                st.download_button(
                    "이 표를 CSV로 저장", shown.to_csv(index=False).encode("utf-8-sig"),  # 엑셀에서 한글이 안 깨지게 BOM 포함
                    file_name=f"분석결과_{d['result_id']}.csv", mime="text/csv", key=f"dl-{idx}-{d['result_id']}",
                )

    with st.expander("계산 과정 보기"):
        for i, s in enumerate(msg["steps"], 1):
            mark = "성공" if s["ok"] else "실패 → 고쳐서 다시 시도"
            st.markdown(f"**{i}. {s['tool']}** ({mark})")
            st.code(", ".join(f"{k}={v}" for k, v in s["args"].items()) or "(입력 없음)", language=None)
            if s["detail"]:
                st.caption(s["detail"])
        u = msg["usage"]
        st.caption(f"LLM 호출 {u['calls']}회 · 토큰 {u['total_tokens']:,}개 (입력 {u['prompt_tokens']:,} / 출력 {u['output_tokens']:,})")


def render_intro(name: str) -> None:
    """데이터 소개: 어떤 데이터인지, 기간, 컬럼 뜻, 전처리에서 바꾼 것. LLM을 부르지 않는다."""
    t, d = TABLES[name], describe_data(name)
    rep = t["report"]
    st.markdown(f"**{t['title']}**  \n{d['rows']:,}행" + (f" · 기간 {d['period']}" if d.get("period") else ""))

    rows = []
    for c in d["columns"]:
        if c["kind"] == "숫자":
            sample = f"{c['min']:,} ~ {c['max']:,}" if isinstance(c.get("min"), (int, float)) else ""
        else:
            values = c.get("values", [])
            sample = ", ".join(map(str, values[:5])) + (" …" if len(values) > 5 else "")
        rows.append({"컬럼": c["column"], "뜻": c["meaning"], "종류": c["kind"].split(" (")[0], "값 예시": sample})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    notes = [f"인코딩: {rep.get('encoding')}"]
    if rep["date"]:
        notes.append(f"날짜 컬럼: {rep['date']['column']} ({rep['date']['unit']} 단위) → 연도·월 컬럼 자동 생성")
    if rep["numeric_columns"]:
        notes.append(f"숫자로 바꾼 컬럼 {len(rep['numeric_columns'])}개")
    for col, info in rep["broken_fixed"].items():
        notes.append(f"'{col}' 깨진 글자 {info['count']}건 정리 (예: {info['examples'][0]})")
    st.caption("전처리: " + " · ".join(notes))
    for w in rep["warnings"]:
        st.caption(f"알림: {w}")


# ---------------------------------------------------------------------------
# 사이드바: 설정 + 데이터
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. 설정")
    # [학습 포인트] type="password" → 화면에 ●●●로 보임. 입력한 키는 이 탭의 메모리에만 있고 파일로 저장하지 않는다.
    key_input = st.text_input("API 키 (Gemini)", type="password",
                              help="비워 두면 .env 파일의 GEMINI_API_KEY를 씁니다")
    model_input = st.text_input("모델 이름", value=DEFAULT_MODEL, help=".env의 GEMINI_MODEL이 기본값")

    st.header("2. 데이터")
    for r in builtin_results:
        if "error" in r:
            st.error(r["error"])
    if not TABLES:
        st.info("등록된 데이터가 없습니다. data 폴더에 CSV를 넣거나 아래에서 올려 주세요.")

    uploaded = st.file_uploader("CSV 올리기", type=["csv"],
                                help="센터에서 내려받은 CSV(UTF-8) 또는 직접 가진 CSV")
    if uploaded is not None:
        file_key = (uploaded.name, uploaded.size)
        if file_key not in st.session_state.uploaded:  # 처음 올린 파일만 등록 (화면이 다시 실행돼도 한 번만)
            name = table_name_from_file(uploaded.name)
            res = register_table(name, uploaded.getvalue(), title=uploaded.name,
                                 source_label=f"이용자 제공 파일 ({uploaded.name})")
            if "error" in res:
                st.error(res["error"])
            else:
                st.session_state.uploaded[file_key] = name
                st.session_state.focus_table = name
                st.success(f"'{uploaded.name}' 등록 완료")

    if TABLES:
        names = list(TABLES)
        st.session_state.focus_table = st.radio(
            "소개를 볼 데이터", names,
            index=names.index(st.session_state.focus_table) if st.session_state.focus_table in names else 0,
            format_func=lambda n: TABLES[n]["title"],
        )


# ---------------------------------------------------------------------------
# 가운데: 데이터 소개 → 예시 질문 → 대화
# ---------------------------------------------------------------------------
st.title("제주 데이터 분석 도우미")
st.caption("CSV를 올리고 일상어로 물어보면, 계산해서 그래프·표·설명으로 답합니다. 데이터로 알 수 없는 질문은 알려 드립니다.")

pending = None  # 예시 버튼을 누르면 여기에 질문이 담김
focus = st.session_state.focus_table
if focus in TABLES:
    with st.expander("데이터 소개", expanded=not st.session_state.messages):
        render_intro(focus)
    examples = example_questions(focus)
    if examples:
        st.caption("예시 질문")
        for col, q in zip(st.columns(len(examples)), examples):
            if col.button(q, key=f"ex-{focus}-{q}"):
                pending = q

# 지난 대화 다시 그리기 (파일이 매번 처음부터 실행되므로)
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            render_assistant(msg, i)

# 새 질문 (입력창 또는 예시 버튼)
question = st.chat_input("예: 2024년 카드 이용 금액이 가장 큰 업종 5개는?") or pending
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("분석 중입니다..."):
            result = run_agent(question, api_key=key_input or None, model=model_input or None)
        msg = {"role": "assistant", **result}
        render_assistant(msg, len(st.session_state.messages))
    st.session_state.messages.append(msg)
