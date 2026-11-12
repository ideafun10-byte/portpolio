"""
LLM에게 지금까지의 대화 내역을 통째로 보내고 답을 받습니다 
"""
import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS

load_dotenv()  # .env의 값을 환경변수로 읽어온다(키를 코드에 쓰지 않기 위해)

MODEL = os.getenv("GEMINI_MODEL")
MAX_STEPS = 6 #사용횟수 제한 



SYSTEM_PROMPT =  """\
너는 제주빅데이터센터 이용자(기관·기업 실무자, 연구자)를 돕는 데이터 분석 도우미다.
이용자는 코딩을 모른다. 일상어 질문을 알아듣고, 보고서에 바로 옮길 수 있게 답한다.
 
[지금 사용할 수 있는 표]
{tables}
 
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
- 아래 형식으로 짧게 쓴다. 출처는 화면이 자동으로 붙이므로 쓰지 않는다.
  **결론** 한두 문장
  **기준** 도구 결과의 basis를 쉬운 말로 (조건, 기간, 집계 방법)
  **한계** 도구 결과의 notes, 데이터 기간 등 이용자가 알아야 할 점 (없으면 생략)
- 데이터로 답할 수 없는 질문은 계산하지 말고 정중히 거절한다.
  표에 없는 정보(예: 지역·연령 컬럼이 없는데 묻는 경우), 데이터 기간 밖, 미래 예측, 원인·이유 판단이 그렇다.
  거절할 때는 이유를 말하고, 이 데이터로 대신 할 수 있는 분석을 하나 제안한다.
  원인을 묻는 질문에는 변화 폭(숫자)은 보여 줄 수 있지만 원인은 판단하지 않는다고 말한다.
- 데이터 분석과 관계없는 요청에는 이 도우미가 할 수 있는 일을 짧게 안내한다.
- 한국어로, 전문 용어 없이 쉽게 쓴다.
"""


def build_system_prompt() -> str:
    """등록된 표마다 이름,제목,행수,기간,컬럼(한글 뜻)을 한줄 씩 적어 프롬프트를 완성한다
    
    
    """
    lines = []
    for name,t in TABLES.item():
        df, date = t["df"],t["report"].get("date")

        period = f", 기간 {df['period'].min()} ~ {df['period'].max()}" if date and "period" in df.columns else ""
        cols = ", ".join(f"{c}({label(c)})" for c in df.columns)
        lines.append(f"- {name}: {t['title']} / {len(df):,}행{period}\n  컬럼: {cols}")
        return SYSTEM_PROMPT_TEMPLATE.format(tables="\n".join(lines) or "(없음)")

    #llm모델은 이 함수 한곳에서만 수행하니 나중에 ai를 바꿀시 여기만 교체하면 된다 
    def call_llm(client,model : str,contents: list,config) :
        return client.models.generate_content(model = model,cotents = contents,config = config)


    def _api_error_message(e :errors.APIError) -> str:
        """api 오류가 났을때 이용자가 이해할 수 있도록  안내 문장을 출력 
        """

        code = getattr(e , "code" ,None)  #e객체의 "code" 필드값 접근
        text = str(e)