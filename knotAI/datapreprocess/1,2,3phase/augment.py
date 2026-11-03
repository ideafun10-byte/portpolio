# -*- coding: utf-8 -*-
"""
데이터 불리기 (증강) 스크립트  v4   (v2 를 고친 버전)

!! 반드시 train/val 분할을 끝낸 뒤, train 폴더만 가리켜서 실행하세요.
   전체 폴더에 증강하면 같은 원본의 사본이 train/val 로 갈려
   val 점수가 실제보다 크게 부풀려집니다. val 은 원본 그대로 둡니다.

v2 대비 바뀐 점:
  [1] 화면 밖으로 나간 점을 가장자리로 밀어넣던 방식을 없애고, 도형 종류별 규칙으로 처리
        - 점(crossing)   : 화면 밖이면 그 증강본을 통째로 버림 (가짜 교차점 방지)
        - 꺾은선(rope)    : 이미지 경계에서 정확히 잘라냄. 중간이 끊기면 버림
        - 사각형(anchor)  : 화면 안쪽 부분만 남김. 거의 안 보이면 그 사각형만 뺌
        - 다각형          : 사각형과 같은 방식
  [2] 사각형은 꼭짓점 4개를 모두 변환한 뒤, 보이는 부분을 감싸는 상자를 다시 구함
  [3] 한글 경로에서도 사진 읽기/쓰기가 되게 하고, 저장 실패를 실제로 확인
  [4] 출력 폴더가 비어 있지 않으면 중단. 같은 이름으로 두 번 저장하려 하면 건너뜀
  [5] 최상위 폴더에서 라벨을 찾는 폴백은 사진 이름이 하나뿐일 때만 사용
  [6] 노이즈 세기를 albumentations 버전과 상관없이 같게 맞춤
  [7] 버전별 인자 이름을 "넣어보고 실패하면 다음" 이 아니라, 직접 확인해서 고름
  [8] 구버전에서도 SEED 가 먹도록 random 모듈까지 고정
  [9] 진행률 계산에 enumerate 사용 (리스트를 매번 처음부터 뒤지지 않음)
  [10] imageData 를 읽자마자 비움 (큰 글자 덩어리를 여러 번 복사하지 않음)

설치:
  pip install -q albumentations
"""

import inspect
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

# ============================================================
# [설정]
# ============================================================
IMG_DIR  = r"C:\knot_work\split_data\train\images"
JSON_DIR = r"C:\knot_work\split_data\train\labels"
OUT_IMG  = r"C:\knot_work\added_data\train\images"
OUT_JSON = r"C:\knot_work\added_data\train\labels"

N_PER_IMG = 5        # 사진 한 장당 몇 장 만들지
KEEP_ORIG = True     # 원본도 결과 폴더에 같이 복사
SEED      = 42       # 같은 숫자면 항상 같은 결과
DRY_RUN   = True    # 먼저 True 로 미리보기 (몇 장 나올지만 확인)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --- 화면 밖으로 나간 도형을 어떻게 할지 ---
IF_POINT_OUT    = "skip"   # "skip" = 그 증강본을 통째로 버림 / "drop" = 그 점만 뺌
EDGE_TOL        = 2.0      # 점이 이 픽셀 이내로 살짝 벗어난 건 가장자리에 붙여서 살림
MIN_VISIBLE     = 0.3      # 사각형·다각형이 이 비율보다 적게 보이면 그 도형은 뺌 (0.3 = 30%)
MIN_LINE_LEN    = 5.0      # 잘리고 남은 로프 조각이 이 픽셀보다 짧으면 없는 것으로 침

# --- 노이즈 세기: 밝기 0~255 기준의 표준편차 범위 ---
#     숫자가 클수록 사진이 거칠어집니다. Over/Under 단서를 지키려고 약하게 둡니다.
NOISE_STD = (5, 15)

# ============================================================


# ------------------------------------------------------------
# 사진 읽기 / 쓰기   [수정 3]
# ------------------------------------------------------------
# 윈도우에서 cv2.imread / cv2.imwrite 는 경로에 한글이 있으면 실패할 수 있고,
# imwrite 는 실패해도 에러 없이 False 만 돌려줍니다.
# 그래서 "파일을 여는 일"은 파이썬(numpy)에게 맡기고,
# OpenCV 는 "사진 내용을 해석/압축하는 일"만 하게 나눴습니다.

def imread_rgb(path):
    """사진을 RGB 배열로 읽습니다. 실패하면 None."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)   # 파일을 통째로 바이트로 읽기
    except OSError:
        return None
    if raw.size == 0:
        return None
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)          # 바이트 -> 사진 배열
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, img):
    """RGB 배열을 사진 파일로 저장합니다. 성공하면 True, 실패하면 False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok, raw = cv2.imencode(path.suffix, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not ok:
            return False
        raw.tofile(str(path))                          # 바이트 -> 파일
    except (OSError, cv2.error):
        return False
    return True


# ------------------------------------------------------------
# 증강 방법 만들기
# ------------------------------------------------------------
def make(cls, new, old, **common):
    """[수정 7] 설치된 버전이 받는 인자 이름을 직접 확인해서 변환을 만듭니다.

    albumentations 2.x 는 모르는 인자를 줘도 에러를 내지 않고 경고만 한 뒤
    기본값으로 만들어 버립니다. 그래서 "넣어보고 에러 나면 다음 걸 시도" 하는
    방식은 엉뚱한 세기의 변환이 조용히 만들어질 위험이 있습니다.
    여기서는 인자 이름이 실제로 있는지부터 확인합니다.

    new    : 신버전 인자 이름으로 쓴 값
    old    : 구버전 인자 이름으로 쓴 값
    common : 두 버전에서 이름이 같은 인자
    """
    params = inspect.signature(cls.__init__).parameters
    for kw in (new, old):
        if all(k in params for k in kw):
            return cls(**kw, **common)
    raise RuntimeError(
        f"{cls.__name__}: 이 버전이 받는 인자 이름을 찾지 못했습니다.\n"
        f"  받는 인자: {[k for k in params if k != 'self']}")


def build_transform(seed=None):
    lo, hi = NOISE_STD

    downscale = make(A.Downscale,
                     {"scale_range": (0.6, 0.9)},
                     {"scale_min": 0.6, "scale_max": 0.9})

    # [수정 6] 신버전은 "최대 밝기(255) 대비 비율", 구버전은 "분산(=표준편차의 제곱)"
    #   으로 세기를 받습니다. 같은 NOISE_STD 에서 각각 계산해 세기 범위를 같게 맞춥니다.
    gauss = make(A.GaussNoise,
                 {"std_range": (lo / 255, hi / 255)},
                 {"var_limit": (lo ** 2, hi ** 2)})

    jpeg = make(A.ImageCompression,
                {"quality_range": (50, 90)},
                {"quality_lower": 50, "quality_upper": 90})

    # 빈 공간(회전하면 생기는 모서리)을 검은색으로 채우는 인자도 버전마다 이름이 다릅니다.
    affine = make(A.Affine,
                  {"border_mode": cv2.BORDER_CONSTANT},
                  {"mode": cv2.BORDER_CONSTANT},
                  scale=(0.85, 1.15),
                  translate_percent=(-0.05, 0.05),
                  rotate=(-30, 30),
                  p=0.8)

    aug = [
        # --- 모양 (점도 같이 움직임) ---
        # 좌우 반전(HorizontalFlip) 금지: 거울상이 되어 pyknotid 단계에서
        #   교차점 부호가 뒤집혀 다른 매듭으로 계산됩니다.
        # 상하 반전(VerticalFlip) 금지: 조명이 아래에서 오는 사진은 현실에 없고,
        #   그림자 방향이 뒤집혀 로프의 위/아래를 잘못 배웁니다.
        affine,

        # --- 색/밝기 (점은 그대로) ---
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                             val_shift_limit=20, p=0.7),
        A.ToGray(p=0.25),
        A.RandomGamma(p=0.3),

        # --- 무늬 뭉개기 (약하게) ---
        # Over/Under 는 교차점에서 어느 가닥의 윤곽선이 끊기지 않는지로 판별하는데,
        # 과한 블러/다운스케일은 그 단서를 지워 "라벨은 over 인데 보이지 않는" 샘플을 만듭니다.
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(3, 5)),
            downscale,
        ], p=0.3),

        A.OneOf([gauss, jpeg], p=0.3),
    ]

    # remove_invisible=False : 화면 밖으로 나간 점도 지우지 않고 돌려받습니다.
    #   (지우면 어느 점이 어느 도형 것이었는지 순서가 어긋납니다. 정리는 rebuild 가 합니다.)
    kp = A.KeypointParams(format="xy", remove_invisible=False)

    # 2.x 는 Compose(seed=) 를 지원합니다. 구버전은 없어서 main 의 random.seed 로 고정합니다.
    if "seed" in inspect.signature(A.Compose.__init__).parameters:
        return A.Compose(aug, keypoint_params=kp, seed=seed)
    return A.Compose(aug, keypoint_params=kp)


# ------------------------------------------------------------
# 도형 다루기   [수정 1] [수정 2]
# ------------------------------------------------------------
def kind_of(shape, n):
    """도형의 종류를 정합니다. n 은 점의 개수."""
    t = str(shape.get("shape_type", "")).lower()
    if n == 1:
        return "point"                     # 교차점
    if t == "rectangle" and n in (2, 4):
        return "box"                       # 앵커(사각형)
    if t in ("linestrip", "line") and n >= 2:
        return "line"                      # 로프(꺾은선)
    if t == "polygon" and n >= 3:
        return "polygon"                   # 다각형
    return "other"                         # 그 밖 (원 등, 전용 규칙 없음)


def flatten(shapes):
    """모든 도형의 점을 한 줄로 폅니다. (증강 라이브러리에 한꺼번에 넘기려고)

    반환:
      pts  : [(x, y), ...]  모든 점
      plan : 도형마다 (종류, pts 에서의 시작 위치, 점 개수, 원래 점 2개짜리 사각형이었는지)

    [수정 2] 점 2개(왼쪽 위, 오른쪽 아래)로 저장된 사각형은 꼭짓점 4개로 늘립니다.
      두 점만 회전시키면 나머지 두 꼭짓점의 정보가 없어 상자가 찌그러지기 때문입니다.
    """
    pts, plan = [], []
    for s in shapes:
        p = [(float(q[0]), float(q[1])) for q in s.get("points", [])]
        kind = kind_of(s, len(p))
        two_pt = False
        if kind == "box" and len(p) == 2:
            (x1, y1), (x2, y2) = p
            p = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            two_pt = True
        plan.append((kind, len(pts), len(p), two_pt))
        pts.extend(p)
    return pts, plan


def _clip_t(p, q, xmax, ymax):
    """선분 p→q 중 화면 안에 있는 구간을 (시작, 끝) 비율로 돌려줍니다. 없으면 None.

    비율 0 은 p 지점, 1 은 q 지점입니다.
    예) (0.0, 0.4) 이면 p 에서 출발해 40% 지점에서 화면 밖으로 나간다는 뜻.
    """
    t0, t1 = 0.0, 1.0
    # 가로(x) 방향과 세로(y) 방향을 각각 검사해서 구간을 좁혀 나갑니다.
    for start, move, limit in ((p[0], q[0] - p[0], xmax), (p[1], q[1] - p[1], ymax)):
        if move == 0:
            if start < 0.0 or start > limit:
                return None                # 이 방향으로 안 움직이는데 이미 밖
        else:
            ta, tb = (0.0 - start) / move, (limit - start) / move
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return None                # 화면과 겹치는 구간이 없음
    return t0, t1


def _between(p, q, t):
    """p 와 q 사이, 비율 t 지점의 좌표"""
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _length(pts):
    """꺾은선의 전체 길이 (픽셀)"""
    return sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
               for a, b in zip(pts[:-1], pts[1:]))


def clip_line(pts, w, h):
    """꺾은선을 이미지 경계에서 잘라, 화면 안에 남는 조각들을 돌려줍니다.

    v2 는 밖으로 나간 점을 가장자리로 "밀어넣어서" 로프가 테두리를 따라 달리는
    가짜 경로가 생겼습니다. 여기서는 로프가 경계선과 만나는 지점을 계산해
    정확히 그 자리에서 자릅니다.

    조각이 1개  -> 정상 (끝부분만 잘림)
    조각이 2개+ -> 로프가 화면 밖으로 나갔다가 다시 들어옴
    조각이 0개  -> 로프가 통째로 화면 밖
    """
    xmax, ymax = w - 1.0, h - 1.0
    runs, cur = [], []
    joined = False                 # 바로 앞 선분의 끝점이 화면 안이었는지

    for p, q in zip(pts[:-1], pts[1:]):
        t = _clip_t(p, q, xmax, ymax)
        if t is None:              # 이 선분은 통째로 화면 밖 -> 조각이 여기서 끊김
            if cur:
                runs.append(cur)
            cur, joined = [], False
            continue

        t0, t1 = t
        a = p if t0 == 0.0 else _between(p, q, t0)     # 화면 안에 들어오는 지점
        b = q if t1 == 1.0 else _between(p, q, t1)     # 화면 밖으로 나가는 지점

        if cur and joined:
            if b != cur[-1]:       # 같은 점이 연달아 들어가지 않게
                cur.append(b)      # 앞 조각에 이어 붙임
        else:
            if cur:
                runs.append(cur)
            cur = [a, b]           # 새 조각 시작
        joined = (t1 == 1.0)

    if cur:
        runs.append(cur)

    # 모서리를 살짝 스친 정도의 아주 짧은 조각은 무시
    return [r for r in runs if _length(r) >= MIN_LINE_LEN]


def clip_polygon(pts, w, h):
    """다각형을 이미지 경계에서 잘라, 화면 안에 남는 부분의 꼭짓점을 돌려줍니다.

    왼쪽 / 오른쪽 / 위 / 아래 경계로 한 번씩, 모두 네 번 잘라냅니다.
    (Sutherland–Hodgman 이라는 방식입니다.)
    """
    xmax, ymax = w - 1.0, h - 1.0
    # (0=가로 좌표 / 1=세로 좌표,  경계값,  경계값보다 큰 쪽이 안쪽인지)
    for axis, limit, keep_bigger in ((0, 0.0, True), (0, xmax, False),
                                     (1, 0.0, True), (1, ymax, False)):
        if not pts:
            break

        def is_in(q):
            return q[axis] >= limit if keep_bigger else q[axis] <= limit

        out = []
        prev = pts[-1]
        for cur in pts:
            if is_in(cur) != is_in(prev):          # 이 변이 경계를 넘나듦
                t = (limit - prev[axis]) / (cur[axis] - prev[axis])
                out.append(_between(prev, cur, t)) #   -> 경계와 만나는 점을 추가
            if is_in(cur):
                out.append(cur)
            prev = cur
        pts = out
    return pts


def _area(pts):
    """다각형의 넓이 (신발끈 공식)"""
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(pts, pts[1:] + pts[:1]))) / 2


def _ring(pts):
    """꼭짓점을 중심 둘레로 한 바퀴 도는 순서로 정렬 (저장 순서가 뒤섞여 있어도 안전하게)"""
    cx = sum(q[0] for q in pts) / len(pts)
    cy = sum(q[1] for q in pts) / len(pts)
    return sorted(pts, key=lambda q: math.atan2(q[1] - cy, q[0] - cx))


def rebuild(shapes, plan, pts, w, h):
    """변환된 점들을 도형별 규칙으로 정리해 새 shapes 리스트를 만듭니다.

    반환: (새 shapes, None, 뺀 도형 집계)   -> 정상
          (None, 버린 이유, None)            -> 이 증강본은 버려야 함
    """
    xmax, ymax = w - 1.0, h - 1.0
    out, dropped = [], Counter()

    def inside(x, y):
        """좌표를 화면 범위 안으로 맞춤 (계산 오차 수준의 미세한 삐져나옴 정리용)"""
        return (min(max(x, 0.0), xmax), min(max(y, 0.0), ymax))

    for s, (kind, start, n, two_pt) in zip(shapes, plan):
        p = [(float(q[0]), float(q[1])) for q in pts[start:start + n]]

        if kind == "point":
            x, y = p[0]
            far_out = (x < -EDGE_TOL or y < -EDGE_TOL or
                       x > xmax + EDGE_TOL or y > ymax + EDGE_TOL)
            if far_out:
                # 가장자리로 밀어넣으면 "교차가 없는 곳에 교차점 라벨" 이 생깁니다.
                if IF_POINT_OUT == "drop":
                    dropped["점"] += 1
                    continue
                return None, "점(교차점)이 화면 밖으로 나감", None
            new_p = [inside(x, y)]

        elif kind == "line":
            runs = clip_line(p, w, h)
            if not runs:
                return None, "로프가 통째로 화면 밖으로 나감", None
            if len(runs) > 1:
                return None, "로프가 화면 밖으로 나갔다 들어와 끊김", None
            new_p = [inside(x, y) for x, y in runs[0]]

        elif kind == "box":
            # 회전된 사각형(꼭짓점 4개) 중 화면 안에 보이는 부분만 잘라낸 뒤,
            # 그 부분을 감싸는 기울지 않은 상자를 다시 구합니다.
            # (회전하면 원래보다 조금 넉넉한 상자가 됩니다. 물체를 놓치지 않는 쪽입니다.)
            quad = _ring(p)
            vis = clip_polygon(quad, w, h)
            full = _area(quad)
            if full <= 0 or len(vis) < 3 or _area(vis) / full < MIN_VISIBLE:
                dropped["사각형"] += 1                          # 거의 안 보이면 라벨도 뺌
                continue
            x1, y1 = inside(min(q[0] for q in vis), min(q[1] for q in vis))
            x2, y2 = inside(max(q[0] for q in vis), max(q[1] for q in vis))
            if two_pt:                                         # 원래 저장 형식 그대로
                new_p = [(x1, y1), (x2, y2)]
            else:
                new_p = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        elif kind == "polygon":
            vis = clip_polygon(p, w, h)
            full = _area(p)
            if full <= 0 or len(vis) < 3 or _area(vis) / full < MIN_VISIBLE:
                dropped["다각형"] += 1
                continue
            new_p = [inside(x, y) for x, y in vis]

        else:
            # 규칙을 만들지 않은 도형 종류: v2 방식(가장자리로 붙이기)을 그대로 씁니다.
            # 이런 도형이 있으면 실행 끝에 경고로 알려줍니다.
            new_p = [inside(x, y) for x, y in p]

        ns = dict(s)                                           # 다른 정보(label 등)는 그대로
        ns["points"] = [[round(x, 2), round(y, 2)] for x, y in new_p]
        out.append(ns)

    return out, None, dropped


# ------------------------------------------------------------
# 저장 / 폴더 검사
# ------------------------------------------------------------
def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def is_inside(child, parent):
    """child 폴더가 parent 폴더와 같거나 그 안에 있는지"""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_out_dirs(pairs):
    """[수정 4] 출력 폴더가 안전한지 검사합니다.  pairs = [(입력, 출력), ...]"""
    for src, out in pairs:
        # 출력이 입력 안에 있으면, 다음에 실행할 때 증강본을 또 증강하게 됩니다.
        if is_inside(out, src) or is_inside(src, out):
            raise SystemExit(f"[중단] 입력과 출력 폴더가 겹칩니다.\n"
                             f"        입력: {src}\n        출력: {out}")
        # 예전 결과가 남아 있으면 새 결과와 섞여도 알 수 없습니다.
        if not DRY_RUN and out.exists() and any(out.iterdir()):
            raise SystemExit(f"[중단] 출력 폴더가 비어 있지 않습니다: {out}\n"
                             "        예전 결과와 섞이지 않게, 폴더를 비우거나\n"
                             "        OUT_IMG / OUT_JSON 을 새 폴더 이름으로 바꿔 주세요.")


# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
def main():
    img_dir, json_dir = Path(IMG_DIR), Path(JSON_DIR)
    out_img, out_json = Path(OUT_IMG), Path(OUT_JSON)

    for p in (img_dir, json_dir):
        if not p.exists():
            q = p
            while not q.exists() and q != q.parent:
                q = q.parent
            raise SystemExit(f"[중단] 경로 없음: {p}\n        존재하는 상위: {q}\n"
                             f"        그 안: {[x.name for x in q.iterdir()][:20]}")

    if IF_POINT_OUT not in ("skip", "drop"):
        raise SystemExit(f'[중단] IF_POINT_OUT 은 "skip" 또는 "drop" 이어야 합니다: {IF_POINT_OUT!r}')

    check_out_dirs([(img_dir, out_img), (json_dir, out_json)])

    print(f"albumentations {A.__version__}")

    # [수정 8] 구버전 albumentations 는 파이썬 기본 random 모듈로 난수를 뽑습니다.
    #   np.random 만 고정하면 재현이 안 되므로 둘 다 고정합니다.
    random.seed(SEED)
    np.random.seed(SEED)
    transform = build_transform(SEED)

    # 하위 폴더(클래스 폴더)까지 훑기
    images = sorted(p for p in img_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)

    if not images:
        raise SystemExit(f"[중단] 사진이 없습니다: {img_dir}\n"
                         "        (train 폴더를 가리키고 있는지 확인하세요)")

    per_class = Counter(str(p.parent.relative_to(img_dir)) for p in images)
    print(f"\n원본 사진: {len(images)}장")
    print("[폴더별]")
    for k, v in sorted(per_class.items()):
        print(f"  {k if k != '.' else '(최상위)'} : {v}장 "
              f"-> 최대 {v * (N_PER_IMG + int(KEEP_ORIG))}장")
    print()

    # [수정 5] 확장자를 뺀 이름이 전체에서 몇 번 나오는지 세어 둡니다.
    stem_count = Counter(p.stem.lower() for p in images)

    # [수정 4] 이번 실행에서 이미 쓴 출력 이름. 같은 이름을 또 쓰려 하면 막습니다.
    #   (윈도우는 대소문자를 구분하지 않으므로 소문자로 통일해서 비교)
    written = set()

    def claim(rel, stem):
        """출력 이름을 '찜' 합니다. 이미 누가 찜했으면 False."""
        key = str(rel / stem).lower()
        if key in written:
            return False
        written.add(key)
        return True

    made, n_orig = 0, 0
    skipped = Counter()          # 버린 증강본: 이유별 개수
    dropped = Counter()          # 증강본은 살리고 도형만 뺀 개수
    other_types = Counter()      # 규칙이 없는 도형 종류
    no_label, read_fail, name_clash = [], [], []
    first_error_shown = False

    # [수정 9] enumerate 로 "지금 몇 번째인지" 를 바로 받습니다.
    for idx, p in enumerate(images, 1):
        if idx % 200 == 0:
            print(f"  ... {idx}/{len(images)}")

        rel = p.parent.relative_to(img_dir)        # 클래스 폴더 유지
        shown = str(rel / p.name)

        # a.jpg 와 a.png 처럼 이름이 같고 확장자만 다르면 라벨(a.json)이 겹칩니다.
        if not claim(rel, p.stem):
            name_clash.append(shown)
            continue

        jp = json_dir / rel / f"{p.stem}.json"
        # [수정 5] 최상위에서 찾는 폴백은 이름이 전체에서 하나뿐일 때만.
        #   여러 클래스에 같은 이름의 사진이 있으면 남의 라벨이 붙기 때문입니다.
        if not jp.exists() and stem_count[p.stem.lower()] == 1:
            jp = json_dir / f"{p.stem}.json"
        if not jp.exists():
            no_label.append(shown)
            continue

        img = imread_rgb(p)
        if img is None:
            read_fail.append(f"{shown} (사진)")
            continue

        try:
            data = json.loads(jp.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):             # 깨졌거나 형식이 다른 JSON
            read_fail.append(f"{shown} (라벨 JSON)")
            continue

        # [수정 10] imageData 에는 사진이 글자로 통째로 들어 있을 수 있습니다.
        #   읽자마자 비워서 아래에서 여러 번 복사되지 않게 합니다.
        #   (키를 지우면 AnyLabeling 이 KeyError 를 내므로 None 으로 둡니다.)
        data["imageData"] = None

        shapes = data.get("shapes") or []
        pts, plan = flatten(shapes)
        for s, (kind, _, _, _) in zip(shapes, plan):
            if kind == "other":
                other_types[str(s.get("shape_type", "?"))] += 1

        if KEEP_ORIG and not DRY_RUN:
            dst = out_img / rel / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

            d0 = dict(data)
            d0["imagePath"] = p.name
            save_json(out_json / rel / f"{p.stem}.json", d0)
        n_orig += 1

        for i in range(N_PER_IMG):
            try:
                r = transform(image=img, keypoints=pts)
            except Exception as e:
                # 원인을 삼키지 않고 처음 한 번은 그대로 보여줍니다.
                if not first_error_shown:
                    print(f"[!] 변환 실패: {e.__class__.__name__}: {e}")
                    first_error_shown = True
                skipped["변환 실패"] += 1
                continue

            new_img, new_pts = r["image"], r["keypoints"]
            h, w = new_img.shape[:2]

            if len(new_pts) != len(pts):           # 점이 사라지면 도형과 짝이 어긋남
                skipped["점 개수가 달라짐"] += 1
                continue

            new_shapes, why, lost = rebuild(shapes, plan, new_pts, w, h)
            if new_shapes is None:
                skipped[why] += 1
                continue

            stem = f"{p.stem}_aug{i}"
            if not claim(rel, stem):
                skipped["출력 이름이 겹침"] += 1
                continue
            name = stem + p.suffix.lower()

            if not DRY_RUN:
                # [수정 3] 사진 저장이 실제로 됐는지 확인하고, 됐을 때만 라벨을 씁니다.
                if not imwrite_rgb(out_img / rel / name, new_img):
                    skipped["사진 저장 실패"] += 1
                    continue

                d = dict(data)                     # 겉만 복사 (shapes 는 새로 만든 것)
                d["shapes"] = new_shapes
                d["imagePath"] = name
                d["imageWidth"], d["imageHeight"] = w, h
                save_json(out_json / rel / f"{stem}.json", d)

            dropped.update(lost)
            made += 1

    # ---------------- 결과 요약 ----------------
    n_skip = sum(skipped.values())
    print("\n" + "=" * 50)
    print("DRY RUN (실제로 저장 안 됨)" if DRY_RUN else "완료")
    print("=" * 50)
    print(f"만든 사진 : {made}장")
    print(f"버린 것   : {n_skip}장")
    for why, n in skipped.most_common():
        print(f"    - {why}: {n}장")
    if KEEP_ORIG:
        print(f"원본 포함 : 총 {made + n_orig}장")

    if dropped:
        print("\n증강본은 살리고 도형만 뺀 것 (화면 밖으로 나가서)")
        for k, n in dropped.items():
            print(f"    - {k}: {n}개")

    if other_types:
        print("\n[!] 전용 규칙이 없는 도형이 있습니다. 화면 밖 점을 가장자리로 붙였습니다.")
        for k, n in other_types.items():
            print(f"    - shape_type={k}: {n}개")

    for title, items in (("라벨 없음", no_label),
                         ("읽기 실패", read_fail),
                         ("이름이 겹쳐 건너뜀 (확장자만 다르거나, 다른 사진의 "
                          "증강본 이름과 같음)", name_clash)):
        if items:
            print(f"\n{title} : {len(items)}장")
            for x in items[:10]:
                print(f"    {x}")

    if n_skip > made * 0.3:
        print("\n[!] 버린 비율이 높습니다. Affine 의 rotate/scale 폭을 줄이거나,\n"
              "    장수를 맞추려면 N_PER_IMG 를 늘려보세요.")

    if DRY_RUN:
        print("\n괜찮으면 DRY_RUN = False 로 바꾸고 다시 실행하세요.")
    else:
        print(f"\n사진 -> {out_img}")
        print(f"라벨 -> {out_json}")


# ============================================================
# 확인용: 결과를 눈으로 보기
# ============================================================
def check(n=3):
    import matplotlib.pyplot as plt

    if os.name == "nt":                            # 윈도우: 제목의 한글이 깨지지 않게
        plt.rcParams["font.family"] = "Malgun Gothic"

    out_img, out_json = Path(OUT_IMG), Path(OUT_JSON)
    files = sorted(q for q in out_img.rglob("*_aug*")
                   if q.suffix.lower() in IMG_EXTS)
    if not files:
        print("결과가 없습니다. DRY_RUN = False 로 main() 을 먼저 실행하세요.")
        return

    # 앞에서부터 n 장만 보면 전부 같은 원본이라, 전체에서 골고루 뽑습니다.
    files = files[::max(1, len(files) // n)][:n]

    fig, axes = plt.subplots(1, len(files), figsize=(6 * len(files), 6))
    if len(files) == 1:
        axes = [axes]

    for ax, p in zip(axes, files):
        img = imread_rgb(p)
        if img is not None:
            ax.imshow(img)

        rel = p.parent.relative_to(out_img)
        jp = out_json / rel / f"{p.stem}.json"
        if jp.exists():
            d = json.loads(jp.read_text(encoding="utf-8"))
            for s in d.get("shapes", []):
                a = np.array(s["points"], dtype=float)
                if a.size == 0:
                    continue
                kind = kind_of(s, len(a))
                if kind == "point":                # 교차점: 빨간 점
                    ax.scatter(a[:, 0], a[:, 1], c="red", s=70,
                               edgecolors="white", linewidths=1.2, zorder=3)
                elif kind == "box":                # 앵커: 노란 상자
                    x1, y1 = a.min(axis=0)
                    x2, y2 = a.max(axis=0)
                    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                            c="yellow", lw=2)
                elif kind == "polygon":            # 다각형: 하늘색 (끝과 처음을 이어 닫음)
                    b = np.vstack([a, a[:1]])
                    ax.plot(b[:, 0], b[:, 1], c="cyan", lw=2)
                else:                              # 로프: 연두색 선
                    ax.plot(a[:, 0], a[:, 1], c="lime", lw=2)
        else:
            ax.set_xlabel("no label")

        ax.set_title(f"{rel}/{p.name}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    if not DRY_RUN:
        check(3)