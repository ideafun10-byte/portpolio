"""
LLM에게 지금까지의 대화 내역을 통째로 보내고 답을 받습니다 
"""
import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors,types

from tools import TABLES, TOOL_FUNCTIONS, get_tool_declarations, label

load_dotenv()  # .env의 값을 환경변수로 읽어온다(키를 코드에 쓰지 않기 위해)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL")
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

        if (code == 400 and "API key" in text) or code in (401,403):
            return "API 키가 올바르지 않거나 권한이 없습니다,왼쪽 [API 키]칸을 확인해주거나 직원에게 문의해주세요 "
        #.env파일의 키를 확인해주세요

        if code == 404 : 
            return "모델의 이름을 찾을 수 없습니다 "
        #.env파일의 GEMINI_MODEL을 확인해주세요

        if code == 429:
            return "LLM의 사용 한도를 넘었습니다 ,1분 뒤쯤에 다시 질문해주세요 "

        if code is not None and code>= 500: #비교를 하기 전에 None 확인
            return "LLM 서버에 일시적인 문제가 있습니다. 잠시 뒤에 다시 질문해 주세요."
        return f"LLM 호출 중 오류가 났습니다 ({code}): {text[:200]}"


def _add_usage(out: dict,response) -> None:
    """응답에 사용한 토큰 사용량을 누적한다.(없으면 0으로 침)"""
    u = getattr(response, "usage_metadata", None)

    out["usage"]["calls"] += 1

    if u is None:
        return 
    out["usage"]["prompt_tokens"] += u.prompt_token_count or 0
    out["usage"]["output_tokens"] += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    out["usage"]["total_tokens"] += u.total_token_count or 0    




def execute_tool(name: str , args : dict) -> dict : 
    """llM이 고른 도구를 실제로 실행한다 , 어떤 경우에도 예외를 던지지 않고 dict를 돌려준다 """

    func = TOOL_FUNCTIONS.get(name)

    if func is None:
        return {"error": f"{name}이라는 도구는 존재하지 않습니다 ,쓸 수 있는 도구 {list(TOOL_FUNCTIONS)}"}

    try:
        return func(**args)

    except TypeError as e : #필요한 값이 빠졋거나 없는 이름을 넘겼을 때 
        return {"error" : f"도구 입력값이 잘못되었습니다 : {e}"}

    except Exception as e:
        return {"error" : f"도구 실행 중 오류 : {type(e).__name__}: {e}"}


def _collect(out : dict , name : str , args : dict, result : dict) -> None:
    """도구 실행기록(steps),화면에 그릴 것(display),출처(sources)를 모은다"""
    ok = "error" not in result
    out["steps"].append({"tool": name , "args" : args, "ok" : ok,
                            "detail" : result.get("error") or result.get("basis") or result.get("message","")})


    if not ok:
        return
    if name == "filter_and_aggregate" and result.get("result_id"):
        out["displays"].append({"type":"table","result_id": result["result_id"],"basis" : result["basis"]})

    if name == 'make_chart':
        out["displays"].append({"type" : "chart", **result["chart"]})

    if result.get("source") and result["source"] not in out["sources"]:
        out["sources"].append(result["source"]) 


def run_agent(question : str, api_key : str | None = None,model : str |None = None) -> dict:
    """질문 하나를 끝까지 처리해 화면에 필요한 것을 돌려준다
    
     돌려주는 dict
      answer    LLM이 쓴 답 (또는 오류 안내 문장)
      displays  [{"type": "table"/"chart", "result_id", ...}]  → app.py가 그림,표 그래프에 들어갈 인자
      steps     [{"tool", "args", "ok", "detail"}]             → '계산 과정 보기'
      sources   ["데이터 출처", ...]                            → 답 아래에 붙임
      usage     {"calls", "prompt_tokens", "output_tokens", "total_tokens"}
      error     None 또는 오류 종류 (화면에서 경고 색으로 보여 줄 때 씀)
    """

    out = {"answer" : "","display": [],"steps": [],"sources": [] , "error": None,
           "usage":{"calls": 0,"prompt_tokens" : 0, "output_tokens" : 0, "total_tokens" : 0}}


    api_key = api_key or os.getenv("GEMINI_API_KEY")
    model = model or DEFAULT_MODEL
    if not api_key:
        out.update(answer = "API 키가 없습니다 ,왼쪽 'API 키'킨에 입력하거나 .env 파일에 키를 작성하세요 ",
                   error = " no_key")
        return out

    if not model:
        out.update(answer = "모델 이름이 없습니다, 왼쪽 '모델 이름'칸이나 ..env 파일을 확인해주세요",
                   error = "no_model")
        return out

    if not TABLES:
        out.update(answer="분석할 데이터가 없습니다. data 폴더에 CSV를 넣거나 왼쪽에서 CSV를 올려 주세요.",
                   error="no_data")
        return out

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(),
        tools = [types.Tool(function_declarations= get_tool_declarations())],
        temperature= 0,  #같은 질문에는 같은 답을 하도록 (검증하기 쉬움)
    )

    contents = [types.Content(role = "user", parts = [types.Part(text = question)])]

    try:
        for _ in range(MAX_STEPS):
            response = call_llm(client, model, contents, config)
            _add_usage(out, response)
 
            # (A) 도구 요청이 없다 = LLM이 답을 다 썼다 → 끝
            if not response.function_calls:
                out["answer"] = (response.text or "").strip() or "답을 만들지 못했습니다. 질문을 조금 바꿔서 다시 해 주세요."
                return out
 
            # (B) LLM의 '도구 요청' 메시지를 기록에 그대로 붙인다
            #     (다음 호출 때 LLM이 자기가 뭘 요청했는지 알도록. 모델의 내부 '생각 서명'도 이 안에 들어 있어서 통째로 붙여야 함)
            contents.append(response.candidates[0].content)

            result_parts = []
            for call in response.function_calls:
                args = dict(call.args or {})
                result = execute_tool(call.name, args)
                _collect(out, call.name,args,result)
                result_parts.append(types.Part.from_function_response(name = call.name,response = result))

            contents.append(types.Content(role = "user",parts = result_parts))

            #사용 상한까지 도구만 부르고 답을 안 쓴 경우 지금까지의 결과로 답하라고 부르는 것 
            #model = "None"으로 설정하고 도구 요청을 금지 ,반드시 글로만 답해야 한다 

        final_config = config.model_copy(update = {"tool_config" : types.ToolConfig(
            function_calling_config= types.FunctionCallingConfig(mode = "NONE")
        )})

        response = call_llm(client,model,contents,final_config)
        _add_usage(out, response)

        out["answer"] = (response.text or "").strip() or "계산 단계가 너무 길어져 답을 끝내지 못했습니다 ,질문을 나눠서 해주세요 "
        return out

    except errors.APIError as e:
        out.update(answer = _api_error_message(e), error = f"api_{getattr(e,'code','')}")

    except Exception as e : #네크워크 끊김 등 다양한 오류 발생
        out.update(answer = f"예상하지 못한 오류가 났습니다 :{type(e).__name__} : {e}",error = "unexpected")
    return out



if __name__ == "__main__" : 
    from tools import load_builtin_tables


    load_builtin_tables()
    q = sys.argv[1] if len(sys.argv) > 1 else "어떤 데이터를 분석 할 수 있어? "
    result = run_agent(q)
    print("답:\n" + result["answer"])
    print("\n출처:", result["sources"])
    print("과정:", json.dumps(result["steps"], ensure_ascii=False, indent=1))
    print("사용량:", result["usage"])