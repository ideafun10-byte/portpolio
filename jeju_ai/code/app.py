"""회면을 담당하는 파일
실행:  streamlit run code/app.py
 
Streamlit에서 꼭 알아야 할 한 가지:
사용자가 뭔가 입력할 때마다 이 파일 전체가 '위에서 아래로 다시' 실행됩니다.
그래서 지난 대화를 기억하려면 st.session_state(새로고침 전까지 유지되는 저장소)에 넣어 둬야 합니다."""

import streamlit as st

from agent import run_agent

st.set_page_config(page_title ="제주 데이터 분석 도우미 에이전트")
st.title("제주 데이터 분석 도우미 에이전트")
st.caption("데이터 출처:Jeju DataHub")

if "messages" not in st.session_state:
    st.session_state.messages = [] # [{"role": "user"/"assistant", "content": ...}, ...]


#1.지난 대화 다시 그리기 (파일이 매번 다시 실행되므로)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # TODO: assistant 메시지에 표·그래프가 있었다면 그것도 다시 그리기
 
# TODO(11/18 개선): 예시 질문 버튼 몇 개


 
# 2) 새 질문 받기
question = st.chat_input("예: 작년 읍면동별 ○○ 추이를 보여 줘")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
 
    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            result = run_agent(question)  # {"answer", "steps", "displays"}
        st.write(result["answer"])
        # TODO: result["displays"]의 표·그래프 그리기 (st.dataframe, st.line_chart / st.bar_chart)
        # TODO: result["steps"]를 st.expander("계산 과정 보기") 안에 보여 주기
 
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})