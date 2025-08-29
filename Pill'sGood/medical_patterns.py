# 의학 관련 패턴들을 관리하는 파일

# 통증 관련 패턴
PAIN_PATTERNS = [
    r'[가이]?\s*[아파|아프|아픔|아픈]',
    r'[가이]?\s*[쓰려|쓰린]',
    r'[가이]?\s*[시큰|시큰거려]'
]

# 불편함 관련 패턴
DISCOMFORT_PATTERNS = [
    r'[가이]?\s*[안\s*좋아|나빠|불편해]',
    r'[가이]?\s*[이상해|거북해]'
]

# 부작용 관련 패턴
SIDE_EFFECT_PATTERNS = [
    r'부작용',
    r'[가이]?\s*[나빠졌어|악화]',
    r'[가이]?\s*[새로\s*생겼어]',
    r'부작용\s*경험',
    r'부작용\s*후기',
    r'부작용\s*경험담',
    r'부작용\s*이야기',
    r'부작용\s*사례',
    r'부작용\s*증상',
    r'부작용\s*문제',
    r'부작용\s*걱정',
    r'부작용\s*피해',
    r'부작용\s*피하고',
    r'부작용\s*없나',
    r'부작용\s*어떤',
    r'부작용\s*심한',
    r'부작용\s*가벼운',
    r'부작용\s*심각한',
    r'부작용\s*나타나',
    r'부작용\s*발생',
    r'부작용\s*생기',
    r'부작용\s*보이',
    r'부작용\s*느끼',
    r'부작용\s*겪',
    r'부작용\s*당하',
    r'부작용\s*걸리',
    r'부작용\s*알려',
    r'부작용\s*알고',
    r'부작용\s*알려줘',
    r'부작용\s*알려주세요',
    r'부작용\s*알려주실',
    r'부작용\s*알려주시면',
    r'부작용\s*알려주시겠어요',
    r'부작용\s*알려주시겠습니까',
    r'부작용\s*알려주세요',
    r'부작용\s*알려주시면',
    r'부작용\s*알려주시겠어요',
    r'부작용\s*알려주시겠습니까'
]

# 경험담 관련 패턴
EXPERIENCE_PATTERNS = [
    r'경험담|후기|경험|사용후기|복용후기',
    r'[가이]?\s*[어땠어|어떠니|쓰고\s*봤어]'
]

# 효능 관련 패턴
EFFICACY_PATTERNS = [
    r'효능|효과|작용|도움',
    r'[가이]?\s*[좋아|나아졌어|좋아졌어|해결됐어]'
]

# 최신 정보 관련 패턴
LATEST_PATTERNS = [
    r'최신|새로|신약|2024|2023',
    r'[가이]?\s*[새로\s*나왔어|최근에]'
]

# 신체 부위 패턴
BODY_PART_PATTERNS = {
    "head": [r'머리|뇌|이마|관자놀이'],
    "stomach": [r'배|속|위|위장'],
    "throat": [r'목|목구멍|인후'],
    "digestion": [r'소화|소화불량'],
    "chest": [r'가슴|심장|폐'],
    "limbs": [r'팔|다리|손|발'],
    "senses": [r'눈|코|귀|입']
}

# 강도 관련 패턴
INTENSITY_PATTERNS = {
    "severe": [r'너무|매우|정말|엄청|심하게|심한|심각한'],
    "mild": [r'가벼운|약한|살짝|조금']
}

# 의도별 검색어
INTENT_SEARCH_TERMS = {
    "pain_relief": ["pain relief", "pain medicine", "analgesic"],
    "discomfort_relief": ["discomfort relief", "medicine for discomfort"],
    "side_effect": ["side effects", "adverse effects", "medicine side effects"],
    "experience_review": ["medicine experience", "drug review", "medication testimonial"],
    "efficacy": ["medicine effectiveness", "drug efficacy", "treatment results"],
    "latest_info": ["new medicine", "latest medication", "recent drug approval"],
    "general_info": ["medicine", "medication", "drug information"]
}

# 부위별 검색어
BODY_PART_SEARCH_TERMS = {
    "head": ["headache medicine", "head pain relief", "migraine treatment"],
    "stomach": ["stomach pain medicine", "abdominal pain relief", "digestive medicine"],
    "throat": ["throat pain medicine", "sore throat relief", "throat medicine"],
    "digestion": ["digestive medicine", "stomach discomfort", "digestion medicine"]
}
