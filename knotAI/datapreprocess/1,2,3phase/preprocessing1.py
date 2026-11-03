# -*- coding: utf-8 -*-
"""
이미지 크기 줄이기 ── 폴더 계층 유지 버전 (v2)

v1 대비 바뀐 점:
  [1] 라벨 배율을 JSON 의 imageWidth/imageHeight 기준으로 계산.
      실제 픽셀 크기와 JSON 선언 크기가 어긋나면(EXIF 회전 불일치 등)
      조용히 틀린 좌표를 쓰지 않고 SKIP 하고 리포트에 남김.
  [2] 줄일 필요 없고 회전도 없는 JPEG 은 재인코딩하지 않고 그대로 복사.
      (불필요한 화질 손실 방지)

사용법:
  아래 [설정] 만 고치고 실행. 처음엔 DRY_RUN = True 로 미리보기.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

# ============================================================
# [설정] 이 부분만 고치세요
# ============================================================

SRC_IMG = r"G:\내 드라이브\knot_dataset\10knots\Knot_my_data\Knot_sm_data"
SRC_JSON = r"G:\내 드라이브\knot_dataset\10knots\Knot_my_data\Knot_sm_data"

OUT_IMG = r"G:\내 드라이브\knot_dataset\10knots\Knot_my_data\knot_pre_sm_data\image"
OUT_JSON = r"G:\내 드라이브\knot_dataset\10knots\Knot_my_data\knot_pre_sm_data\label"

MAX_SIDE = 1280      # 긴 변을 이 크기로. 0 이면 크기 안 줄이고 정리만.
QUALITY = 92         # JPEG 품질 (90~95 권장)

DRY_RUN = False      # True 면 실제로 저장 안 하고 미리보기만

# 라벨 JSON 을 어떻게 찾을지
#   "same" : 사진과 같은 하위 폴더 안에서 찾기 (기본)
#   "flat" : SRC_JSON 바로 아래에 다 모여 있음
#   "any"  : 어디 있든 이름으로 전체 검색
JSON_MODE = "same"

# 종횡비가 이 비율 이상 어긋나면 좌표계 불일치로 보고 SKIP
ASPECT_TOL = 0.02

# ============================================================

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
EXIF_ORIENTATION = 0x0112


def find_json(stem: str, rel_parent: Path, src_json: Path):
    """라벨 파일 찾기 — 여러 방식을 순서대로 시도"""
    if JSON_MODE in ("same", "any"):
        c = src_json / rel_parent / f"{stem}.json"
        if c.exists():
            return c
    if JSON_MODE in ("flat", "any", "same"):
        c = src_json / f"{stem}.json"
        if c.exists():
            return c
    if JSON_MODE == "any":
        hits = list(src_json.rglob(f"{stem}.json"))
        if hits:
            return hits[0]
    return None


def scale_label(jp: Path, dst_j: Path, dst_i: Path,
                real_wh, new_wh, fallback_r):
    """
    [수정 1] 라벨 좌표 스케일링.

    배율은 이미지 픽셀 크기가 아니라 JSON 이 선언한 크기를 기준으로 계산한다.
    좌표의 원래 기준은 라벨링 당시 화면에 보이던 크기(= JSON 의 imageWidth/Height)
    이지 지금 디스크에서 읽은 픽셀 크기가 아니기 때문.

    반환: ("ok", 메시지) / ("skip", 이유)
    """
    d = json.loads(jp.read_text(encoding="utf-8"))
    w, h = real_wh
    nw, nh = new_wh

    jw = d.get("imageWidth")
    jh = d.get("imageHeight")

    if jw and jh:
        # 가로/세로가 정확히 뒤집힌 경우 = EXIF 좌표계 불일치.
        # 단순 곱셈으로 고칠 수 없으므로 건드리지 않는다.
        if (jw, jh) == (h, w) and jw != jh:
            return ("skip",
                    f"좌표계 불일치(회전): JSON {jw}x{jh} vs 실제 {w}x{h}")

        if abs(jw / jh - w / h) > ASPECT_TOL:
            return ("skip",
                    f"종횡비 불일치: JSON {jw}x{jh} vs 실제 {w}x{h}")

        rx, ry = nw / jw, nh / jh
    else:
        # 크기 정보가 없는 옛 JSON → 이미지 기준 배율로 폴백
        rx = ry = fallback_r

    for s in d.get("shapes", []):
        s["points"] = [[float(x) * rx, float(y) * ry]
                       for x, y in s.get("points", [])]

    # 이미지가 다른 루트에 저장되므로 상대경로로 연결해 둔다
    try:
        rel_path = dst_i.resolve().relative_to(dst_j.resolve().parent)
        d["imagePath"] = str(rel_path).replace("\\", "/")
    except ValueError:
        import os
        d["imagePath"] = os.path.relpath(
            str(dst_i), str(dst_j.parent)).replace("\\", "/")

    d["imageWidth"], d["imageHeight"] = nw, nh
    d.pop("imageData", None)          # 용량 큰 base64 제거

    if not DRY_RUN:
        dst_j.parent.mkdir(parents=True, exist_ok=True)
        dst_j.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    return ("ok", f"rx={rx:.4f} ry={ry:.4f}")


def main():
    src_img = Path(SRC_IMG)
    src_json = Path(SRC_JSON)
    out_img = Path(OUT_IMG)
    out_json = Path(OUT_JSON)

    if not src_img.exists():
        print(f"[중단] 폴더가 없습니다: {src_img}")
        print("       Drive 마운트를 했는지, 경로가 맞는지 확인하세요.")
        return

    images = sorted(p for p in src_img.rglob("*")
                    if p.is_file()
                    and p.suffix.lower() in EXTS
                    and not p.name.startswith("."))

    if not images:
        print(f"[중단] 사진을 찾지 못했습니다: {src_img}")
        return

    print(f"사진 {len(images)}장 발견\n")

    folders = Counter(str(p.parent.relative_to(src_img)) for p in images)
    print("[폴더별 사진 수]")
    for k, v in sorted(folders.items()):
        print(f"  {k if k != '.' else '(최상위)'} : {v}장")
    print()

    done = 0
    copied = 0            # [수정 2] 재인코딩 없이 그대로 복사한 수
    labeled = 0
    nolabel, skipped, failed = [], [], []
    size_before, size_after = 0, 0
    sample = None

    for p in images:
        rel_parent = p.parent.relative_to(src_img)
        dst_i = out_img / rel_parent / f"{p.stem}.jpg"

        try:
            with Image.open(p) as im:
                src_mode = im.mode
                orient = im.getexif().get(EXIF_ORIENTATION, 1)
                im = ImageOps.exif_transpose(im)     # 회전을 픽셀에 반영
                if im.mode != "RGB":
                    im = im.convert("RGB")

                w, h = im.size
                if MAX_SIDE and max(w, h) > MAX_SIDE:
                    r = MAX_SIDE / max(w, h)
                    nw, nh = round(w * r), round(h * r)
                    im = im.resize((nw, nh), Image.LANCZOS)
                else:
                    r, nw, nh = 1.0, w, h

                # ---- [수정 2] 손댈 게 없으면 재인코딩하지 않고 복사 ----
                passthrough = (
                    r == 1.0
                    and orient in (0, 1)                    # 회전 없음
                    and src_mode == "RGB"                   # 변환 없음
                    and p.suffix.lower() in (".jpg", ".jpeg")
                )

                if not DRY_RUN:
                    dst_i.parent.mkdir(parents=True, exist_ok=True)
                    if passthrough:
                        shutil.copy2(p, dst_i)
                        copied += 1
                    else:
                        im.save(dst_i, "JPEG", quality=QUALITY,
                                subsampling=0, optimize=True)
                    size_after += dst_i.stat().st_size

            size_before += p.stat().st_size
            if sample is None:
                sample = (p.name, (w, h), (nw, nh), r, passthrough)

        except Exception as e:
            failed.append((p.name, f"{e.__class__.__name__}: {e}"))
            continue

        # ---- 라벨 ----
        jp = find_json(p.stem, rel_parent, src_json)
        if jp is None:
            nolabel.append(str(rel_parent / p.name))
        else:
            dst_j = out_json / rel_parent / f"{p.stem}.json"
            try:
                status, msg = scale_label(jp, dst_j, dst_i,
                                          (w, h), (nw, nh), r)
                if status == "ok":
                    labeled += 1
                else:
                    skipped.append((str(rel_parent / jp.name), msg))
            except Exception as e:
                failed.append((jp.name, f"라벨처리실패: {e}"))

        done += 1
        if done % 100 == 0:
            print(f"  ... {done}/{len(images)}")

    # ---- 결과 ----
    print("\n" + "=" * 55)
    print("DRY RUN (실제로 저장 안 됨)" if DRY_RUN else "완료")
    print("=" * 55)
    print(f"처리한 사진   : {done}장")
    print(f"  재인코딩    : {done - copied}장")
    print(f"  그대로 복사 : {copied}장  (줄일 필요 없어 화질 보존)")
    print(f"변환한 라벨   : {labeled}개")

    if sample:
        n, (w, h), (nw, nh), r, pt = sample
        tag = " [복사]" if pt else ""
        print(f"크기 변화     : {w}x{h} -> {nw}x{nh}  (배율 {r:.3f}){tag}")

    if not DRY_RUN and size_before:
        print(f"용량          : {size_before/1024/1024:.0f} MB "
              f"-> {size_after/1024/1024:.0f} MB")

    if skipped:
        print(f"\n[!] 좌표계 불일치로 건너뛴 라벨 : {len(skipped)}개")
        print("    아래 파일은 좌표가 변환되지 않았습니다. 확인 필요.")
        for n, m in skipped[:15]:
            print(f"    {n}\n        {m}")
        if len(skipped) > 15:
            print(f"    ... 외 {len(skipped)-15}개")

    if nolabel:
        print(f"\n라벨 없는 사진 : {len(nolabel)}장 (아직 라벨링 안 했으면 정상)")
        for x in nolabel[:10]:
            print(f"    {x}")
        if len(nolabel) > 10:
            print(f"    ... 외 {len(nolabel)-10}장")
        print("  -> 라벨이 다른 폴더에 있으면 JSON_MODE 를 'any' 로 바꿔보세요.")

    if failed:
        print(f"\n실패 : {len(failed)}건")
        for n, e in failed[:10]:
            print(f"    {n} : {e}")

    if not DRY_RUN:
        print("\n[결과 폴더 구조]")
        for d in sorted(out_img.rglob("*")):
            if d.is_dir():
                cnt = len([x for x in d.iterdir() if x.is_file()])
                print(f"  {d.relative_to(out_img)}/ : {cnt}장")

    if DRY_RUN:
        print("\n결과가 괜찮으면 DRY_RUN = False 로 바꾸고 다시 실행하세요.")


# ============================================================
# 검증 1: 전체 파일의 정규화 좌표가 보존됐는지 대조 (숫자로)
# ============================================================
def verify_all(tol=0.002):
    src_json, out_json = Path(SRC_JSON), Path(OUT_JSON)
    bad, checked = [], 0

    for jp_new in out_json.rglob("*.json"):
        rel = jp_new.relative_to(out_json)
        jp_old = src_json / rel
        if not jp_old.exists():
            jp_old = src_json / rel.name
            if not jp_old.exists():
                continue

        a = json.loads(jp_old.read_text(encoding="utf-8"))
        b = json.loads(jp_new.read_text(encoding="utf-8"))
        aw, ah = a.get("imageWidth"), a.get("imageHeight")
        bw, bh = b.get("imageWidth"), b.get("imageHeight")
        if not all([aw, ah, bw, bh]):
            continue

        checked += 1
        hit = False
        for sa, sb in zip(a.get("shapes", []), b.get("shapes", [])):
            for (x1, y1), (x2, y2) in zip(sa.get("points", []),
                                          sb.get("points", [])):
                if (abs(x1 / aw - x2 / bw) > tol or
                        abs(y1 / ah - y2 / bh) > tol):
                    bad.append(str(rel))
                    hit = True
                    break
            if hit:
                break

    print(f"검증 {checked}개 — 이상 {len(bad)}건")
    for x in bad[:20]:
        print("   ", x)


# ============================================================
# 검증 2: 눈으로 보기
# ============================================================
def check(n=3):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    out_img, out_json = Path(OUT_IMG), Path(OUT_JSON)
    cands = [p for p in sorted(out_img.rglob("*.jpg"))
             if (out_json / p.parent.relative_to(out_img) /
                 f"{p.stem}.json").exists()]
    files = cands[:n] or sorted(out_img.rglob("*.jpg"))[:n]
    if not files:
        print("결과가 없습니다. DRY_RUN = False 로 main() 을 먼저 실행하세요.")
        return

    fig, axes = plt.subplots(1, len(files), figsize=(7 * len(files), 7))
    if len(files) == 1:
        axes = [axes]

    for ax, p in zip(axes, files):
        ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))

        rel = p.parent.relative_to(out_img)
        jp = out_json / rel / f"{p.stem}.json"
        if jp.exists():
            d = json.loads(jp.read_text(encoding="utf-8"))
            for s in d.get("shapes", []):
                a = np.array(s["points"])
                if a.size == 0:
                    continue
                lab = s.get("label", "").lower()
                if "crossing" in lab:
                    ax.scatter(a[:, 0], a[:, 1], c="red", s=90,
                               edgecolors="white", linewidths=1.5, zorder=3)
                elif "anchor" in lab:
                    ax.plot(*np.append(a, a[:1], axis=0).T, c="yellow", lw=2)
                else:
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
        verify_all()
        check(3)