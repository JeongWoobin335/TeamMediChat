from qa_state import QAState
from answer_utils import generate_response_llm, extract_field
from langchain_core.documents import Document
import re

def contains_exact_product_name(doc: Document, product_name: str) -> bool:
    return re.search(rf"\[제품명\]:\s*{re.escape(product_name)}\b", doc.page_content) is not None

def generate_final_answer_node(state: QAState) -> QAState:
    # ✅ 이미 final_answer가 설정된 경우 (최신 정보 요청 등)
    if state.get("final_answer"):
        return state

    # ✅ 병력 기반 추천이 있는 경우 먼저 반환하고 종료
    if state.get("recommendation_answer"):
        state["final_answer"] = state["recommendation_answer"]
        return state

    fields = state.get("requested_fields") or ["효능", "부작용", "사용법"]
    relevant_docs = state.get("relevant_docs") or []

    if relevant_docs:
        name = extract_field(relevant_docs, "제품명") or "이 약"
        eff = extract_field(relevant_docs, "효능", name)
        side = extract_field(relevant_docs, "부작용", name)
        usage = extract_field(relevant_docs, "사용법", name)

        # 🎯 모든 요청 필드에 대해 적절한 fallback 구성
        field_values = {}
        for field in fields:
            if field in ["효능", "부작용", "사용법"]:
                value = extract_field(relevant_docs, field, name)
                if value != "정보 없음":
                    field_values[field] = value
            else:
                # 비표준 필드는 사용법으로 fallback
                if usage != "정보 없음":
                    field_values[field] = f"(사용법 참조) {usage}"

        answer = generate_response_llm(
            name,
            list(field_values.keys()),
            field_values.get("효능", "정보 없음"),
            field_values.get("부작용", "정보 없음"),
            field_values.get("사용법", "정보 없음")
        )

        state["final_answer"] = answer
        return state



    if state.get("external_parsed"):
        data = state["external_parsed"]
        name = data.get("제품명") or "이 약"
        answer = generate_response_llm(
            name,
            fields,
            data.get("효능", "정보 없음"),
            data.get("부작용", "정보 없음"),
            data.get("사용법", "정보 없음")
        )
        state["final_answer"] = answer
        return state

    state["final_answer"] = "죄송합니다. 해당 약품에 대한 정보를 찾을 수 없습니다."
    return state
