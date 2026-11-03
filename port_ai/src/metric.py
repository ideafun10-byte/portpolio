"""
대회 점수 함수,내가 평가하기 위해 사용하는 것으로 공모전에 제출 하지 않을 것
"""
#names:기록의 선박명
#y_true:정답 배 ID
#y_pred:내 모델이 추측한 ID
import re
from src.normalize import normalize_name   # 수정: normalize.py 안의 실제 함수 이름과 맞추기


def weighted_accuracy(names, y_true, y_pred):
    """0.7 × 한글 정확도 + 0.3 × 영문 정확도.
    돌려주는 값: (가중 점수, 한글 정확도, 영문 정확도)"""

    # 수정: 입력을 모두 '문자열 리스트'로 통일
    #  - 리스트로 바꾸면 Series여도 i번째 = 위치로 동작 (라벨 문제 없음)
    #  - str()로 바꾸면 "00123"과 123 같은 자료형 차이로 틀리는 일이 없음
    names  = list(names)
    y_true = [str(v) for v in y_true]
    y_pred = [str(v) for v in y_pred]

    K_list = []   # 한글 기록의 위치
    F_list = []   # 한글이 아닌 기록의 위치
    for i, name in enumerate(names):
        # 수정: 빈칸(NaN)은 문자열이 아니므로 빈 문자열로 보고 '한글 아님'으로 분류
        text = normalize_name(name) if isinstance(name, str) else ""
        if re.search("[가-힣]", text) is None:
            F_list.append(i)
        else:
            K_list.append(i)

    # 수정: 더하기 전에 시작값 0을 먼저 만들어 둠
    K_score = 0
    F_score = 0
    for i in K_list:
        if y_true[i] == y_pred[i]:
            K_score += 1
    for i in F_list:
        if y_true[i] == y_pred[i]:
            F_score += 1

    # 수정: 그룹이 비어 있으면 0으로 나누지 않도록 정확도를 0으로 둠
    K_acc = K_score / len(K_list) if K_list else 0.0
    F_acc = F_score / len(F_list) if F_list else 0.0

    total = 0.7 * K_acc + 0.3 * F_acc

    # 수정: 한글·영문 정확도도 같이 돌려줌 (어느 쪽이 올랐는지 보려고)
    return total, K_acc, F_acc