# -*- coding: utf-8 -*-
"""이미지/라벨이 이미 분리된 구조에서 샘플 뽑아 테스트 데이터 만들기"""

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# [설정] — 원본
SRC_IMG  = r"G:/내 드라이브/knot_dataset/10knots/Knot_my_data/knot_pre_sm_data/image"
SRC_JSON = r"G:/내 드라이브/knot_dataset/10knots/Knot_my_data/knot_pre_sm_data/label"

# [설정] — 결과
OUT_IMG  = r"G:/내 드라이브/knot_dataset/10knots/Knot_my_data/test_sample/image"
OUT_JSON = r"G:/내 드라이브/knot_dataset/10knots/Knot_my_data/test_sample/label"

N = 15                 # 뽑을 개수
PER_CLASS = False      # True = 하위 폴더마다 N개씩 / False = 전체에서 총 N개
KEEP_TREE = True       # 하위 폴더 구조 유지
SEED = 42              # 같은 값이면 항상 같은 샘플
DRY_RUN = False         # 먼저 True 로 미리보기

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ============================================================

src_img, src_json = Path(SRC_IMG), Path(SRC_JSON)
out_img, out_json = Path(OUT_IMG), Path(OUT_JSON)

for p in (src_img, src_json):
    if not p.exists():
        q = p
        while not q.exists() and q != q.parent:
            q = q.parent
        raise SystemExit(f"[중단] 경로 없음: {p}\n        존재하는 상위: {q}\n"
                         f"        그 안: {[x.name for x in q.iterdir()][:20]}")

# --- 이미지 기준으로 같은 상대경로의 라벨을 짝지음 ---
pairs = defaultdict(list)
missing = []

for img in sorted(src_img.rglob("*")):
    if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
        continue
    rel = img.relative_to(src_img)
    jp = src_json / rel.with_suffix(".json")
    if not jp.exists():                       # 평탄 구조 대비 폴백
        jp = src_json / f"{img.stem}.json"
    if not jp.exists():
        missing.append(str(rel))
        continue
    pairs[str(rel.parent)].append((img, jp))

total = sum(len(v) for v in pairs.values())
print(f"짝 맞은 이미지 {total}장 / 라벨 못 찾음 {len(missing)}장\n")
print("[폴더별 후보 수]")
for k in sorted(pairs):
    print(f"  {k if k != '.' else '(최상위)'} : {len(pairs[k])}장")
if missing:
    print(f"\n[!] 라벨 없는 이미지 예시: {missing[:5]}")
print()

if total == 0:
    raise SystemExit("[중단] 짝을 하나도 못 찾았습니다. 폴더 구조를 확인하세요.")

# --- 샘플링 ---
rng = random.Random(SEED)
picked = []

if PER_CLASS:
    for k in sorted(pairs):
        pool = pairs[k][:]
        rng.shuffle(pool)
        if len(pool) < N:
            print(f"  [!] {k}: {len(pool)}장뿐이라 전부 사용")
        picked += [(k, *x) for x in pool[:N]]
else:
    pools = {k: rng.sample(v, len(v)) for k, v in pairs.items()}
    keys = sorted(pools)
    while len(picked) < N and any(pools.values()):
        for k in keys:
            if pools[k]:
                picked.append((k, *pools[k].pop()))
                if len(picked) >= N:
                    break
    if len(picked) < N:
        print(f"  [!] 후보가 {total}장뿐이라 {len(picked)}장만 뽑았습니다.")

print(f"뽑힌 수 : {len(picked)}장")
print("[클래스별]")
for k, c in sorted(Counter(k for k, _, _ in picked).items()):
    print(f"  {k} : {c}장")

# --- 복사 ---
manifest = []
for key, img, jp in picked:
    rel = Path(key) if KEEP_TREE and key != "." else Path(".")
    stem = img.stem if KEEP_TREE else f"{key.replace(os.sep, '_')}__{img.stem}"

    dst_i = out_img / rel / f"{stem}{img.suffix.lower()}"
    dst_j = out_json / rel / f"{stem}.json"
    manifest.append(f"{key}/{img.name}")

    if DRY_RUN:
        continue

    dst_i.parent.mkdir(parents=True, exist_ok=True)
    dst_j.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dst_i)

    d = json.loads(jp.read_text(encoding="utf-8-sig"))
    d["imageData"] = None
    d["imagePath"] = os.path.relpath(dst_i, dst_j.parent).replace("\\", "/")
    dst_j.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                     encoding="utf-8")

print("\n" + "=" * 50)
print("DRY RUN (복사 안 됨)" if DRY_RUN else "복사 완료")
print("=" * 50)
for m in manifest:
    print("  ", m)

if not DRY_RUN:
    (out_img.parent / "manifest.txt").write_text(
        "\n".join(manifest), encoding="utf-8")
    print(f"\n이미지 -> {out_img}")
    print(f"라벨   -> {out_json}")
    print("목록   -> manifest.txt")
else:
    print("\n괜찮으면 DRY_RUN = False 로 바꾸고 다시 실행하세요.")

