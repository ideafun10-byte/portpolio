"""
agent.py — 에이전트 루프 (이 프로젝트의 심장)

흐름
  질문 → (인사·잡담이면 준비된 안내로 바로 답) → [LLM이 판단 → 도구 실행 → 결과를 LLM에게 돌려줌]을 반복 → 최종 답

LLM은 대화의 '기억'이 없습니다. 그래서 매번 지금까지의 대화 기록(contents)을
통째로 다시 보냅니다. 이 기록에 '질문 → 도구 요청 → 도구 결과 → ...'가 차례로 쌓입니다.

이 파일이 지키는 약속
- run_agent()는 어떤 경우에도 예외를 밖으로 던지지 않는다 → 화면(app.py)이 멈추지 않음 (MVP 기준 5)
- 질문당 LLM 호출 수와 토큰을 기록한다 → 운영 비용 추정 (MVP 기준 9)
- 출처는 LLM이 아니라 코드가 모아서 붙인다 → 빠뜨릴 일이 없음 (MVP 기준 6)

터미널 테스트:  python agent.py "2024년 카드 이용 금액이 가장 큰 업종 5개는?"
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

from dotenv import dotenv_values
from google import genai
from google.genai import errors, types

from tools import TABLES, TOOL_FUNCTIONS, get_tool_declarations, label

# ---------------------------------------------------------------------------
# [수정 1] 설정(API 키·모델 이름) 정하기
#   우선순위: ① 화면에 입력한 값 → ② .env 파일 → ③ 컴퓨터 환경변수 → ④ 기본값(모델만)
# ---------------------------------------------------------------------------
# [학습 포인트] .env 위치를 '이 파일 옆'으로 못 박아 둔다.
#   위치를 안 적으면 '프로그램을 실행한 폴더'에 따라 .env를 찾기도 하고 못 찾기도 한다.
ENV_PATH = Path(__file__).parent.parent / ".env"
FALLBACK_MODEL = "gemini-3.8-flash"  # .env에도 화면에도 모델 이름이 없을 때 쓰는 기본값
# 보통 질문 하나 = describe_data → filter_and_aggregate → make_chart → 답 쓰기 = 4번.

# 도구가 error를 돌려줘서 한두 번 다시 부르는 경우까지 고려해 6번. (무한 반복·비용 폭주 방지용 상한)
MAX_STEPS = 10


def _read_env_file() -> dict:
    """.env 파일을 '지금' 읽어 {이름: 값} dict로 돌려준다. 파일이 없거나 못 읽으면 빈 dict.

      예전의 load_dotenv()는 프로그램이 켜질 때 한 번만 읽어 os.environ(환경변수 저장소)에 넣었다.
      그리고 이미 들어 있는 이름은 덮어쓰지 않는다. → 앱을 켜 둔 채 .env를 고쳐도 옛 값이 그대로 쓰였다.
      dotenv_values()는 환경변수 저장소를 건드리지 않고 '파일 내용만' dict로 돌려준다.
      질문할 때마다 읽어도 몇 줄짜리 파일이라 부담이 없고, .env를 고치면 바로 반영된다.
    """
    if not ENV_PATH.exists():
        return {}
    # utf-8-sig: 파일 맨 앞에 보이지 않는 표시 글자(BOM)가 있어도 없어도 읽힘
    # cp949:     메모장에서 'ANSI'로 저장한 파일. 한글 주석이 들어 있으면 UTF-8로는 읽다가 오류가 난다
    #            (예전 load_dotenv()는 이 경우 프로그램이 켜지다가 UnicodeDecodeError로 멈췄다)
    for enc in ("utf-8-sig", "cp949"):
        try:
            return {k: v for k, v in dotenv_values(ENV_PATH, encoding=enc).items() if v is not None}
        except UnicodeDecodeError:
            continue
    return {}


def _clean(value) -> str:
    """값 앞뒤의 공백·따옴표를 뗀다. None이면 빈 글자. (복사해 붙이다 딸려 온 공백·줄바꿈 제거)"""
    return str(value or "").strip().strip("\"'").strip()


def _looks_like_key(key: str) -> bool:
    """진짜 키처럼 생겼는지 본다: 영문·숫자·기호만 있고 공백이 없어야 한다.

    [학습 포인트] API 키는 요청의 '머리말(header)'에 실려 가는데, 머리말에는 한글을 넣을 수 없다.
      .env.example의 '여기에_키_붙여넣기'를 그대로 두면 서버에 가기도 전에 UnicodeEncodeError가 난다.
      그래서 보내기 전에 걸러서 '키가 없다'고 알기 쉽게 안내한다.
    """
    return bool(key) and key.isascii() and not any(ch.isspace() for ch in key)


def resolve_settings(api_key: str | None = None, model: str | None = None) -> dict:
    """API 키와 모델 이름을 어디서 가져올지 한곳에서 정한다.

    돌려주는 dict
      api_key      쓸 키 (없으면 "")
      key_source   키를 가져온 곳: "화면 입력" / ".env 파일" / "환경변수" / None
      key_problem  형식이 잘못돼 건너뛴 키가 있으면 그 설명 (없으면 None)
      model        쓸 모델 이름 (항상 값이 있음)
      model_source 모델 이름을 가져온 곳 (위 세 가지 또는 "기본값")
      env_file     .env 파일이 있는가
    """
    file_env = _read_env_file()
    out = {"api_key": "", "key_source": None, "key_problem": None,
           "model": FALLBACK_MODEL, "model_source": "기본값", "env_file": ENV_PATH.exists()}

    for source, value in (("화면 입력", api_key), (".env 파일", file_env.get("GEMINI_API_KEY")),
                          ("환경변수", os.environ.get("GEMINI_API_KEY"))):
        value = _clean(value)
        if not value:
            continue
        if _looks_like_key(value):
            out.update(api_key=value, key_source=source)
            break
        if out["key_problem"] is None:  # 예시 문구 그대로이거나 한글·공백이 섞인 값 → 쓰지 않고 다음 후보로
            out["key_problem"] = f"{source}의 API 키는 예시 문구 그대로이거나 한글·공백이 섞여 있어 쓰지 않았습니다."

    for source, value in (("화면 입력", model), (".env 파일", file_env.get("GEMINI_MODEL")),
                          ("환경변수", os.environ.get("GEMINI_MODEL"))):
        value = _clean(value)
        if value:
            out.update(model=value, model_source=source)
            break
    return out


def settings_status(api_key: str | None = None, model: str | None = None) -> dict:
    """화면(app.py)에 '지금 어떤 설정이 쓰이는지' 보여 주기 위한 요약. 키 자체는 돌려주지 않는다."""
    s = resolve_settings(api_key, model)
    return {"has_key": bool(s["api_key"]), "key_source": s["key_source"], "key_problem": s["key_problem"],
            "model": s["model"], "model_source": s["model_source"],
            "env_file": s["env_file"], "env_path": str(ENV_PATH)}


DEFAULT_MODEL = resolve_settings()["model"]  # 예전 코드와의 호환용 (프로그램이 켜질 때의 기본 모델)


# ---------------------------------------------------------------------------
# [수정 2] 인사·감사·잡담은 LLM을 부르지 않고 바로 답한다 (호출 0회 = 비용 0, 답도 항상 같음)
# ---------------------------------------------------------------------------
# [학습 포인트] 아래 패턴은 re.fullmatch로 검사한다 = 문장 '전체'가 패턴과 맞아야 한다.
#   "안녕"                          → 전체가 인사 → 여기서 바로 답함
#   "안녕, 2024년 매출 알려 줘"     → 뒤에 질문이 붙어 있어 전체가 맞지 않음 → LLM에게 넘어감 (질문을 놓치지 않음)
#   ( ... )+ 는 '괄호 안이 한 번 이상 반복'  → "안녕하세요 반갑습니다"처럼 인사가 이어져도 맞음
#   ? 는 '앞의 것이 있어도 되고 없어도 됨'   → 고마워 / 고마워요
_SMALL_TALK = [
    ("인사", re.compile(r"(안녕(하세요|하십니까|하신가요|하세용|하셔요)?|하이|헬로우?|hi|hello|hey|ㅎㅇ"
                        r"|반가워요?|반갑습니다|좋은(아침|저녁|하루)(이에요|입니다)?)+")),
    ("감사", re.compile(r"(고마워요?|고맙습니다|감사(합니다|해요|드립니다|드려요)?|땡큐|thanks?|thankyou|ㄱㅅ"
                        r"|수고(했어요?|하셨습니다|하세요)?)+")),
    ("작별", re.compile(r"(잘가요?|잘있어요?|안녕히(계세요|가세요)|바이(바이)?|bye(bye)?|다음에(또)?(봐요?|만나요))+")),
    ("소개", re.compile(r"(넌|너는|너|당신은)?누구(야|니|세요|예요|인가요)?"
                        r"|(넌|너는|너)?(뭐|뭘|무엇을|무슨일을?|어떤일을?)"
                        r"(할수있(어|니|나요|어요|습니까)|해줄수있(어|니|나요|어요)|도와줄수있(어|니|나요|어요))"
                        r"|도움말|사용법|사용방법|help|어떻게(써|쓰는거야|사용해|사용하나요)")),
]

_HOW_TO_ASK = ('궁금한 것을 평소 말투로 물어보시면 됩니다. '
               '예: "가장 큰 항목 5개가 뭐야?", "월별로 어떻게 변했는지 그래프로 보여 줘", "어떤 데이터가 있어?"')


def _simplify(text: str) -> str:
    """판별용으로 문장을 단순하게 만든다: 소문자로, 공백·문장부호·이모지 제거, 끝의 ㅋㅋ·ㅎㅎ·ㅠㅠ 제거."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = re.sub(r"[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ]", "", t)  # 한글·영문·숫자만 남김
    return re.sub(r"[ㅋㅎㅠㅜ]+$", "", t)


def classify_small_talk(question: str) -> str | None:
    """인사·감사·작별·소개 요청·빈 말 중 하나면 그 종류를, 데이터 질문일 수 있으면 None을 돌려준다."""
    t = _simplify(question)
    if not t:
        rest = re.sub(r"[ㅋㅎㅠㅜ]", "", str(question))
        return None if any(ch.isalpha() or ch.isdigit() for ch in rest) else "빈말"
    for kind, pattern in _SMALL_TALK:
        if pattern.fullmatch(t):
            return kind
    return None


def _data_line() -> str:
    """지금 등록된 데이터를 한 줄로 소개한다. (제목은 5개까지만 → 답이 길어지지 않게)"""
    if not TABLES:
        return "지금은 등록된 데이터가 없습니다. 왼쪽에서 CSV를 올려 주시면 바로 분석할 수 있습니다."
    titles = [t["title"] for t in TABLES.values()]
    shown = ", ".join(f"'{t}'" for t in titles[:5]) + (f" 외 {len(titles) - 5}개" if len(titles) > 5 else "")
    return f"지금 볼 수 있는 데이터는 {len(titles)}개입니다: {shown}"


def small_talk_reply(question: str) -> str | None:
    """인사·잡담이면 준비된 답을, 아니면 None을 돌려준다. LLM을 부르지 않는다."""
    kind = classify_small_talk(question)
    if kind is None:
        return None
    if kind == "감사":
        return "도움이 되었다니 다행입니다. 더 궁금한 점이 있으면 이어서 물어보세요."
    if kind == "작별":
        return "이용해 주셔서 감사합니다. 필요할 때 다시 질문해 주세요."
    if kind == "빈말":
        return f"질문을 알아듣지 못했습니다. {_HOW_TO_ASK}"
    if kind == "소개":
        return ("저는 표 데이터(CSV)를 대신 계산해 드리는 분석 도우미입니다. "
                "합계·평균·순위·비중·기간별 변화를 계산해서 표와 그래프로 보여 드립니다. "
                "데이터에 없는 내용, 미래 예측, 원인 판단은 하지 않습니다.\n\n"
                f"{_data_line()}\n\n{_HOW_TO_ASK}")
    return f"안녕하세요! 제주 데이터를 대신 계산해 드리는 분석 도우미입니다.\n\n{_data_line()}\n\n{_HOW_TO_ASK}"


# ---------------------------------------------------------------------------
# 프롬프트 (제출 필수 항목: 'Prompt' → 한곳에 모아 둠)
# [학습 포인트] {tables} 자리에는 '지금 등록된 표 목록'이 질문할 때마다 채워진다.
#   CSV를 새로 올리면 LLM이 바로 그 표를 알게 하려고, 고정 문장이 아니라 틀(template)로 둔다.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """\
너는 제주빅데이터센터 이용자(기관·기업 실무자, 연구자)를 돕는 데이터 분석 도우미다.
이용자는 코딩을 모른다. 일상어 질문을 알아듣고, 보고서에 바로 옮길 수 있게 답한다.

[지금 사용할 수 있는 표]
아래 표는 모두 이미 불러와 둔 것이다. 이용자가 '저장된 데이터', '내장 데이터', '올린 파일'이라고 하면 이 목록을 뜻한다.
{tables}
{focus}
- 도구의 name에는 각 줄 맨 앞의 '표 이름'을 글자 그대로 넣는다. (제목이 아니라 이름)
- 이용자가 표를 말하지 않았으면 제목과 컬럼 뜻을 보고 질문에 맞는 표를 고른다.
  맞는 표가 여러 개라 정할 수 없을 때만 어느 표인지 되묻는다.
- 목록에 없는 데이터를 물으면 없다고 말하고, 지금 있는 표의 제목을 알려 준다.

[먼저 질문의 종류를 가린다]
가. 인사·감사·잡담, 도우미 자신에 대한 질문 (예: "안녕", "고마워", "넌 누구야", "뭐 할 수 있어?")
    → 도구를 부르지 않는다. 한두 문장으로 자연스럽게 답하고, 지금 있는 표로 물어볼 만한 질문을 하나 권한다.
나. 어떤 데이터·컬럼이 있는지 묻는 질문
    → 위 표 목록만 보고 답한다. 값 목록까지 필요할 때만 describe_data를 부른다.
다. 숫자 계산이 필요한 데이터 질문 → [일하는 순서]와 [답변 규칙]을 따른다.
라. 데이터로 답할 수 없는 질문, 데이터 분석과 관계없는 요청(날씨, 번역, 일반 상식 등)
    → 계산하지 않고 정중히 거절한다. ([답변 규칙]의 거절 방법)
인사와 질문이 함께 오면 인사는 한마디로 받고 질문을 처리한다.
가·나·라에는 '결론·기준·한계' 틀을 쓰지 않는다.

[일하는 순서]
1. 질문에 맞는 표를 고른다. 거를 값(업종 이름 등)이나 컬럼이 확실하지 않으면 describe_data를 먼저 부른다.
2. filter_and_aggregate로 계산한다. filters의 값은 describe_data의 values에 적힌 글자 그대로 쓴다.
   도구가 error를 돌려주면 내용(비슷한 값 힌트 등)을 읽고 고쳐서 다시 부른다.
3. 시간에 따른 변화는 period로 묶고 make_chart(kind=line), 항목 비교·순위는 make_chart(kind=bar)를 부른다.
   숫자 하나로 끝나는 답에는 그래프를 그리지 않는다.
4. 도구는 꼭 필요한 만큼만 부른다.

[답변 규칙]
- 숫자는 도구 결과에 있는 값만 쓴다. 지어내거나 어림하지 않는다.
  두 값의 비율처럼 간단한 계산이 꼭 필요하면 계산식(값 ÷ 값)을 함께 적는다.
- 큰 금액은 읽기 쉽게 억·만 단위를 함께 적는다. (예: 1,234,567,890원 → 약 12.3억 원)
- 도구로 계산한 답은 아래 틀로 짧게 쓴다. 출처는 화면이 자동으로 붙이므로 쓰지 않는다.
  **결론** 한두 문장
  **기준** 도구 결과의 basis를 쉬운 말로 (조건, 기간, 집계 방법)
  **한계** 도구 결과의 notes, 데이터 기간 등 이용자가 알아야 할 점 (없으면 생략)
- 데이터로 답할 수 없는 질문은 계산하지 말고 정중히 거절한다.
  표에 없는 정보(예: 지역·연령 컬럼이 없는데 묻는 경우), 데이터 기간 밖, 미래 예측, 원인·이유 판단이 그렇다.
  거절할 때는 이유를 말하고, 이 데이터로 대신 할 수 있는 분석을 하나 제안한다.
  원인을 묻는 질문에는 변화 폭(숫자)은 보여 줄 수 있지만 원인은 판단하지 않는다고 말한다.
- 한국어로, 전문 용어 없이 쉽게 쓴다.
"""


def build_system_prompt(focus_table: str | None = None) -> str:
    """등록된 표마다 이름·제목·행 수·기간·컬럼(한글 뜻)을 한 줄씩 적어 프롬프트를 완성한다.

    focus_table: 이용자가 화면에서 보고 있는 표. '이 데이터'가 무엇을 가리키는지 LLM에게 알려 준다.

    [학습 포인트] 컬럼 목록을 미리 알려 주면 LLM이 describe_data를 덜 불러도 된다 → 호출 수·토큰 절약.
      단, 범주 '값' 목록(업종 이름 등)은 길어서 여기 넣지 않고 필요할 때 describe_data로 본다.
    """
    lines = []
    for name, t in TABLES.items():
        df, date = t["df"], t["report"].get("date")
        period = ""
        if date and "period" in df.columns:
            p = df["period"].dropna()  # 날짜로 못 바꾼 빈 값은 빼고 처음·끝을 구함
            if not p.empty:
                period = f", 기간 {p.min()} ~ {p.max()}"
        cols = ", ".join(f"{c}({label(c)})" for c in df.columns)
        lines.append(f"- {name}: {t['title']} / {len(df):,}행{period}\n  컬럼: {cols}")
    focus = ""
    if focus_table in TABLES:
        focus = (f"이용자가 지금 화면에서 보고 있는 표: {focus_table}. "
                 "'이 데이터', '이 표', '지금 데이터'라고 하면 이 표를 뜻한다.")
    return SYSTEM_PROMPT_TEMPLATE.format(tables="\n".join(lines) or "(없음)", focus=focus)


# ---------------------------------------------------------------------------
# LLM 호출은 이 함수 한 곳에서만 → 나중에 다른 회사 API로 바꿀 때 여기만 고치면 됨
# ---------------------------------------------------------------------------
def call_llm(client, model: str, contents: list, config) -> types.GenerateContentResponse:
    return client.models.generate_content(model=model, contents=contents, config=config)


def _api_error_message(e: errors.APIError) -> str:
    """API 오류 코드 → 이용자가 이해할 수 있는 안내 문장."""
    code = getattr(e, "code", None)
    text = str(e)
    if (code == 400 and "API key" in text) or code in (401, 403):
        return "API 키가 올바르지 않거나 권한이 없습니다. 왼쪽 'API 키' 칸이나 .env 파일의 키를 확인해 주세요."
    if code == 404:
        return "모델 이름을 찾을 수 없습니다. 왼쪽 '모델 이름' 칸이나 .env의 GEMINI_MODEL을 확인해 주세요."
    if code == 429:
        return "LLM 사용 한도를 넘었습니다. 1분쯤 뒤에 다시 질문해 주세요."
    if code and code >= 500:
        return "LLM 서버에 일시적인 문제가 있습니다. 잠시 뒤에 다시 질문해 주세요."
    return f"LLM 호출 중 오류가 났습니다 ({code}): {text[:200]}"


def _add_usage(out: dict, response) -> None:
    """응답에 적힌 토큰 사용량을 누적한다. (없으면 0으로 침)"""
    u = getattr(response, "usage_metadata", None)
    out["usage"]["calls"] += 1
    if u is None:
        return
    out["usage"]["prompt_tokens"] += u.prompt_token_count or 0
    out["usage"]["output_tokens"] += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    out["usage"]["total_tokens"] += u.total_token_count or 0


def execute_tool(name: str, args: dict) -> dict:
    """LLM이 고른 도구를 실제로 실행한다. 어떤 경우에도 예외를 밖으로 던지지 않고 dict를 돌려준다."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"'{name}'이라는 도구는 없습니다. 쓸 수 있는 도구: {list(TOOL_FUNCTIONS)}"}
    try:
        return func(**args)
    except TypeError as e:  # 필요한 값이 빠졌거나 없는 이름의 값을 넘긴 경우
        return {"error": f"도구 입력값이 잘못됐습니다: {e}"}
    except Exception as e:  # 그 밖의 모든 오류 → LLM이 읽고 다른 방법을 찾게
        return {"error": f"도구 실행 중 오류: {type(e).__name__}: {e}"}


def _collect(out: dict, name: str, args: dict, result: dict) -> None:
    """도구 실행 기록(steps), 화면에 그릴 것(displays), 출처(sources)를 모은다."""
    ok = "error" not in result
    out["steps"].append({"tool": name, "args": args, "ok": ok,
                         "detail": result.get("error") or result.get("basis") or result.get("message", "")})
    if not ok:
        return
    if name == "filter_and_aggregate" and result.get("result_id"):
        out["displays"].append({"type": "table", "result_id": result["result_id"], "basis": result["basis"]})
    if name == "make_chart":
        out["displays"].append({"type": "chart", **result["chart"]})
    if result.get("source") and result["source"] not in out["sources"]:
        out["sources"].append(result["source"])


def run_agent(question: str, api_key: str | None = None, model: str | None = None,
              focus_table: str | None = None) -> dict:
    """질문 하나를 끝까지 처리해 화면에 필요한 것들을 돌려준다.

    돌려주는 dict
      answer    LLM이 쓴 답 (또는 준비된 안내·오류 안내 문장)
      displays  [{"type": "table"/"chart", "result_id", ...}]  → app.py가 그림
      steps     [{"tool", "args", "ok", "detail"}]             → '계산 과정 보기'
      sources   ["데이터 출처", ...]                            → 답 아래에 붙임
      usage     {"calls", "prompt_tokens", "output_tokens", "total_tokens"}
      error     None 또는 오류 종류 (화면에서 경고 색으로 보여 줄 때 씀)
    """
    out = {"answer": "", "displays": [], "steps": [], "sources": [], "error": None,
           "usage": {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

    # 0) 인사·잡담이면 준비된 답으로 끝 (LLM 호출 0회 → API 키가 없어도 대답함)
    quick = small_talk_reply(question)
    if quick is not None:
        out["answer"] = quick
        return out

    # 1) 시작 전 점검: 하나라도 없으면 LLM을 부르지 않고 바로 안내 (비용 0)
    settings = resolve_settings(api_key, model)
    if not settings["api_key"]:
        reason = settings["key_problem"] or "API 키가 없습니다."
        out.update(answer=f"{reason} 왼쪽 'API 키' 칸에 입력하거나 .env 파일에 GEMINI_API_KEY를 적어 주세요.",
                   error="no_key")
        return out
    if not TABLES:
        out.update(answer="분석할 데이터가 없습니다. data 폴더에 CSV를 넣거나 왼쪽에서 CSV를 올려 주세요.",
                   error="no_data")
        return out

    try:
        # [수정] 준비 단계(프롬프트·도구 설명서 만들기)도 try 안으로 → 여기서 오류가 나도 화면이 멈추지 않음
        client = genai.Client(api_key=settings["api_key"])
        config = types.GenerateContentConfig(
            system_instruction=build_system_prompt(focus_table),
            tools=[types.Tool(function_declarations=get_tool_declarations())],
            temperature=0,  # 같은 질문엔 최대한 같은 답 (검증하기 쉬움)
        )
        contents = [types.Content(role="user", parts=[types.Part(text=question)])]  # 대화 기록

        for _ in range(MAX_STEPS):
            response = call_llm(client, settings["model"], contents, config)
            _add_usage(out, response)

            # (A) 도구 요청이 없다 = LLM이 답을 다 썼다 → 끝
            if not response.function_calls:
                out["answer"] = (response.text or "").strip() or "답을 만들지 못했습니다. 질문을 조금 바꿔서 다시 해 주세요."
                return out

            # (B) LLM의 '도구 요청' 메시지를 기록에 그대로 붙인다
            #     (다음 호출 때 LLM이 자기가 뭘 요청했는지 알도록. 모델의 내부 '생각 서명'도 이 안에 들어 있어서 통째로 붙여야 함)
            contents.append(response.candidates[0].content)

            # (C) 요청받은 도구를 하나씩 실행하고, 결과를 모아 기록에 붙인다
            result_parts = []
            for call in response.function_calls:
                if not call.name: 
                    continue
                args = dict(call.args or {})
                result = execute_tool(call.name, args)
                _collect(out, call.name, args, result)
                result_parts.append(types.Part.from_function_response(name=call.name, response=result))
            contents.append(types.Content(role="user", parts=result_parts))
            # → 다시 for문 처음으로: LLM이 도구 결과를 읽고 다음 행동을 정함

        # (D) 상한까지 도구만 부르고 답을 안 쓴 경우: '이제 도구 없이 지금까지 결과로 답하라'고 한 번 더 부름
        # [학습 포인트] mode="NONE" = 이번 호출에서는 도구 요청을 금지 → 반드시 글로 답함
        final_config = config.model_copy(update={"tool_config": types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE"))})
        response = call_llm(client, settings["model"], contents, final_config)
        _add_usage(out, response)
        out["answer"] = (response.text or "").strip() or "계산 단계가 너무 길어져 답을 끝내지 못했습니다."
        return out

    except errors.APIError as e:
        out.update(answer=_api_error_message(e), error=f"api_{getattr(e, 'code', '')}")
    except Exception as e:  # 네트워크 끊김 등 예상 못 한 오류도 화면을 멈추지 않게
        out.update(answer=f"예상하지 못한 오류가 났습니다: {type(e).__name__}: {e}", error="unexpected")
    return out


if __name__ == "__main__":
    from tools import ensure_builtin_tables

    ensure_builtin_tables()
    q = sys.argv[1] if len(sys.argv) > 1 else "어떤 데이터를 분석할 수 있어?"
    result = run_agent(q)
    print("답:\n" + result["answer"])
    print("\n출처:", result["sources"])
    print("과정:", json.dumps(result["steps"], ensure_ascii=False, indent=1))
    print("사용량:", result["usage"])