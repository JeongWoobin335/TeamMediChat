from qa_state import QAState

def remember_previous_context_node(state: QAState) -> QAState:
    """
    동일 세션 내에서 직전 응답이 있다면 이를 previous_context로 저장합니다.
    이는 follow-up 질문 시 문맥 정보를 보조로 활용하기 위함입니다.
    """
    final_answer = state.get("final_answer")
    if final_answer:
        state["previous_context"] = final_answer
    else:
        state["previous_context"] = None

    return state

