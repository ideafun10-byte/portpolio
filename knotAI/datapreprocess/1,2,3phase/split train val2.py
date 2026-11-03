# -*- coding: utf-8 -*-
"""
[1단계] train / val 나누기  ── 폴더 계층 유지

하는 일:
  1. 매듭 종류 폴더마다 따로 8:2 로 나눔 (모든 종류가 양쪽에 골고루 들어가게)
  2. 사진과 라벨 JSON 을 짝지어서 같이 복사
  3. 원본 폴더 구조를 그대로 유지

주의:
  test 는 이 코드로 만들지 않습니다. 직접 찍은 사진을 따로 보관하세요.

사용법:
  [설정] 만 고치고 실행.
  처음엔 DRY_RUN = True 로 미리보기.
"""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

# ============================================================
# [설정] 이 부분만 고치세요
# ============================================================

SRC_IMG  = r"G:\내 드라이브\knot_dataset\10knots\knot_labeling_image_data\preprocessing_data\pre_knot_image_data"
SRC_JSON = r"G:\내 드라이브\knot_dataset\10knots\knot_labeling_image_data\preprocessing_data\pre_knot_label_data"

OUT_ROOT = r"G:\내 드라이브\knot_dataset\10knots\knot_labeling_image_data\split_data"

VAL_RATIO = 0.2      # val 비율 (0.2 = 20%)
SEED      = 42       # 같은 숫자면 항상 같게 나뉨

DRY_RUN   = False     # True 면 실제로 복사 안 하고 미리보기만

# 라벨 JSON 찾는 방식
#   "same" : 사진과 같은 하위 폴더 구조
#   "flat" : SRC_JSON 바로 아래에 다 모여 있음
#   "any"  : 어디 있든 이름으로 전체 검색
JSON_MODE = "same"

# ============================================================


def find_json(stem: str, rel_parent: Path, src_json: Path):
    if JSON_MODE in ("same", "any"):
        c = src_json / rel_parent / f"{stem}.json"
        if c.exists():
            return c
    c = src_json / f"{stem}.json"
    if c.exists():
        return c
    if JSON_MODE == "any":
        hits = list(src_json.rglob(f"{stem}.json"))
        if hits:
            return hits[0]
    return None


def main():
    src_img  = Path(SRC_IMG)
    src_json = Path(SRC_JSON)
    out      = Path(OUT_ROOT)

    if not src_img.exists():
        print(f"[중단] 폴더가 없습니다:\n  {src_img}")
        return

    exts = (".jpg", ".jpeg", ".png")
    images = sorted(p for p in src_img.rglob("*")
                    if p.is_file()
                    and p.suffix.lower() in exts
                    and not p.name.startswith("."))

    if not images:
        print(f"[중단] 사진을 찾지 못했습니다:\n  {src_img}")
        return

    print(f"사진 {len(images)}장 발견\n")

    # --- 매듭 종류(하위 폴더)별로 묶기 ---
    groups = {}
    for p in images:
        rel = p.parent.relative_to(src_img)
        groups.setdefault(rel, []).append(p)

    rng = random.Random(SEED)
    plan = {"train": [], "val": []}

    print(f"{'폴더':30s} {'전체':>6s} {'train':>7s} {'val':>6s}")
    print("-" * 53)

    for rel in sorted(groups.keys(), key=str):
        files = sorted(groups[rel])
        rng.shuffle(files)

        n_val = max(1, round(len(files) * VAL_RATIO)) if len(files) > 1 else 0
        plan["val"].extend((rel, p) for p in files[:n_val])
        plan["train"].extend((rel, p) for p in files[n_val:])

        name = str(rel) if str(rel) != "." else "(최상위)"
        print(f"{name:30s} {len(files):6d} {len(files)-n_val:7d} {n_val:6d}")

    print("-" * 53)
    print(f"{'합계':30s} {len(images):6d} "
          f"{len(plan['train']):7d} {len(plan['val']):6d}\n")

    # --- 실제 복사 ---
    copied = Counter()
    nolabel = []

    for split, items in plan.items():
        for rel, p in items:
            di = out / split / "images" / rel
            dj = out / split / "labels" / rel

            jp = find_json(p.stem, rel, src_json)
            if jp is None:
                nolabel.append(str(rel / p.name))

            if not DRY_RUN:
                di.mkdir(parents=True, exist_ok=True)
                dj.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, di / p.name)
                if jp is not None:
                    # imageData(용량 큰 base64) 제거하며 복사
                    try:
                        d = json.loads(jp.read_text(encoding="utf-8"))
                        d.pop("imageData", None)
                        (dj / f"{p.stem}.json").write_text(
                            json.dumps(d, ensure_ascii=False, indent=2),
                            encoding="utf-8")
                    except Exception:
                        shutil.copy2(jp, dj / jp.name)

            copied[split] += 1

    # --- 결과 ---
    print("=" * 53)
    print("DRY RUN (실제로 복사 안 됨)" if DRY_RUN else "복사 완료")
    print("=" * 53)
    for k, v in copied.items():
        print(f"  {k} : {v}장")

    if nolabel:
        print(f"\n라벨 없는 사진 : {len(nolabel)}장")
        for x in nolabel[:10]:
            print(f"    {x}")
        if len(nolabel) > 10:
            print(f"    ... 외 {len(nolabel)-10}장")
        print("  -> 라벨이 다른 곳에 있으면 JSON_MODE 를 'any' 로 바꿔보세요.")

    if not DRY_RUN:
        print(f"\n저장 위치: {out}")
        print("\n[만들어진 구조]")
        for split in ("train", "val"):
            base = out / split / "images"
            if base.exists():
                n = len(list(base.rglob("*.*")))
                print(f"  {split}/images/  : {n}장")
                print(f"  {split}/labels/  : "
                      f"{len(list((out/split/'labels').rglob('*.json')))}개")

    if DRY_RUN:
        print("\n결과가 괜찮으면 DRY_RUN = False 로 바꾸고 다시 실행하세요.")


if __name__ == "__main__":
    main()