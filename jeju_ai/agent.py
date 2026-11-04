"""
LLM에게 지금까지의 대화 내역을 통째로 보내고 답을 받습니다 
"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS

load_dotenv()  # .env의 값을 환경변수로 읽어온다(키를 코드에 쓰지 않기 위해)

MODEL = os.getenv("GEMINI_MODEL")
MAX_STEPS = 4 #사용횟수 제한 



SYSTEM_PROMPT = """
너는 제주 공공데이터를 분석해 주는 도우미다.
TODO: 아래 같은 규칙을 직접 정해서 채우기
- 숫자는 반드시 도구 결과에서만 가져온다 (지어내지 않기)
- 컬럼 이름을 모르면 먼저 describe_data를 부른다
- 답 끝에 데이터 출처를 적는다
- 데이터로 답할 수 없는 질문이면 솔직하게 말한다

agent.py의 SYSTEM_PROMPT TODO 자리에 직접 채우실 내용의 뼈대입니다.

이 데이터가 다루는 범위: 무엇(버스 승하차, 교통량), 어디(제주 전체·정류소), 언제(기간), 단위(시간대·일·월)
용어 사전: "붐빈다·사람이 많다" → 승차 인원, "출근 시간" → 7~9시, "퇴근 시간" → 17~19시, "주말" → 토·일
시간 범위는 직접 정한 기준이니, 답에서 "출근 시간(7~9시 기준)"처럼 밝히게 하세요.
기본 가정: 기간이나 지역을 말하지 않으면 '전체 기간·제주 전체'로 계산하고, 답에 그 가정을 적는다.
매번 되묻지 않는 이유가 있습니다. 질문 하나당 LLM 호출 상한이 4회라서, 되물으면 한 번이 낭비됩니다. 되묻기는 ④처럼 핵심 정보가 아예 없을 때만 합니다.
숫자 규칙: 숫자는 도구 결과에서만 가져온다. 도구가 error나 '데이터 없음'을 돌려주면 추측하지 말고 알 수 없다고 한다.
답변 형식(일반인용):
결론 한 줄
그래프
기준(어떤 데이터, 기간, 지표)
한계(예: 버스 이용자만 포함)
출처
컬럼 이름이나 함수 이름 같은 코드 용어는 답에 쓰지 않는다.
"""


def run_agent(question : str) -> dict:
    """질문 하나늘 끝까지 처리해 필요한 인자를 리턴"""

    client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

    config  = types.GenerateContentConfig(
        system_instruction= SYSTEM_PROMPT,
        tools = [types.Tool(function_declarations=TOOL_DECLARATIONS)],
        temperature= 0 #같은 질문에는 같은 답을 하도록 한다 
    )

    contents = [types.Content(role = "user",parts = [types.Part(text= question)])] #대화 기록을 리스트로 반환

    steps = []  #화면의 '계산 과정'에 쓸 기록: 어떤 도구를 어떤 값으로 불렀나?

    displays = [] #화면에 그릴 표,그래프



    for _ in range(MAX_STEPS):
        response = client.models.generate_content(model = MODEL,contents= contents,config = config)
        #LLM에다가 명령을 입력하고 나온 대답은 response에 저장
        print(response)

        #LLM의 도구 요청이 없었다 => 답을 다 작성했다(종료)
        if not response.function_call:
            return {"answer": response.text,"steps" : steps,"display":displays}


        #response.candidate:LLM이 낸 답안 목록
        contents.append(response.candidates[0].content)

         # (C) 요청받은 도구를 하나씩 실행하고, 결과를 모아 기록에 붙인다
        result_parts = []
        
        #response.function_calls:도구 요청 조각만 모아서 리스트에 넣은 것
        for call in response.function_calls:
            args = dict(call.args or {})
            result = execute_tool(call.name,args)
            steps.append({"tool" : call.name , "args":args ,"ok": "error" not in result}) 

            #TODO: result 안에 "chart"(또는 화면에 보여 줄 표)가 있으면 displays에 모으기

            result_parts.append(types.Part.from_function_response(name=call.name, response=result))
        contents.append(types.Content(role="user", parts=result_parts))
        # → 다시 for문 처음으로: LLM이 도구 결과를 읽고 다음 행동을 정함
 
    # TODO: MAX_STEPS를 다 쓸 때까지 답이 안 나온 경우, 사용자에게 무엇을 보여 줄지 정하기
    raise NotImplementedError

def execute_tool(name:str,arg : dict) -> dict:
    """LLM이 고른 도구를 실제로 실행한다. 어떤 경우에도 예외를 밖으로 던지지 않고 dict를 돌려준다."""
    # TODO 1: TOOL_FUNCTIONS에 없는 이름이면 {"error": ...}
    # TODO 2: 함수 실행 중 예외가 나면 {"error": ...}  (앱이 멈추지 않게)
    raise NotImplementedError



if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "어떤 데이터를 분석할 수 있어?"

    out = run_agent(q)
    