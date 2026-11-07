"""
convert_raw.py — 원본 파일 형식 통일 (전처리 전 단계)

하는 일
  data/ 아래의 하위 폴더(예: data/raw, data/hub)마다
  그 안의 CSV·XLSX를 전부 읽어서
  → 인코딩을 UTF-8(BOM 포함, 'utf-8-sig')로 통일한 CSV로
  → data/<폴더이름>_utf8/ 에 저장합니다.

원본은 절대 덮어쓰지 않습니다. (잘못 변환돼도 원본에서 다시 시작할 수 있게)

실행 (code/ 폴더에서):  python scripts/convert_raw.py
필요 패키지: pandas, openpyxl (xlsx 읽기용 → requirements.txt에 추가할 것)

그다음 prepare_data.py는 data/*_utf8/ 의 CSV만 'utf-8-sig'로 읽으면 됩니다.
"""
from pathlib import Path

import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent   # scripts/ 의 한 칸 위 = code/
DATA_DIR = ROOT / "data"
OUT_SUFFIX = "_utf8"  # 결과 폴더 이름 끝에 붙는 글자 (원본 폴더와 구분하는 표시)

# CSV 인코딩을 시도할 순서.
# UTF-8을 먼저: UTF-8은 규칙이 엄격해서 틀리면 바로 오류가 남 → 확실히 걸러짐.
# cp949를 먼저 하면 UTF-8 파일이 오류 없이 '깨진 글자'로 읽힐 수 있음.
ENCODINGS = ["utf-8-sig", "cp949"]


def read_csv_any_encoding(path: Path) -> tuple[pd.DataFrame, str]:
    """ENCODINGS를 차례로 시도해서 읽히는 인코딩으로 CSV를 읽는다."""
    for enc in ENCODINGS:
        try:
            # dtype=str: 모든 값을 '글자 그대로' 읽음.
            #   지역코드 '0501' 같은 값이 숫자 501로 바뀌어 앞의 0이 사라지는 걸 막음.
            #   숫자 변환은 다음 단계(prepare_data.py)에서 컬럼별로 판단해서 함.
            # keep_default_na=False: 빈칸을 NaN으로 바꾸지 않고 '' 그대로 둠 (원본 보존)
            df = pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
            return df, enc
        except UnicodeDecodeError:
            continue  # 이 인코딩이 아니면 다음 후보로
    raise ValueError(f"인코딩을 알 수 없음 (시도: {ENCODINGS})")


def safe_name(text: str) -> str:
    """파일 이름에 쓸 수 없는 문자(Windows 기준)를 _로 바꾼다."""
    for ch in '\\/:*?"<>|':
        text = text.replace(ch, "_")
    return text.strip()


def convert_csv(path: Path, out_dir: Path) -> list[str]:
    df, enc = read_csv_any_encoding(path)
    out = out_dir / f"{path.stem}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")  # index=False: 행 번호 열을 새로 만들지 않음
    return [f"{out.name}  ({enc} → utf-8-sig, {len(df):,}행 × {df.shape[1]}열)"]


def convert_xlsx(path: Path, out_dir: Path) -> list[str]:
    # sheet_name=None: 모든 시트를 {시트이름: DataFrame} 딕셔너리로 읽음
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, engine="openpyxl")
    results = []
    for sheet_name, df in sheets.items():
        if df.empty:
            results.append(f"(시트 '{sheet_name}' 비어 있어서 건너뜀)")
            continue
        df = df.fillna("")  # 엑셀 빈칸 → '' (CSV 쪽과 맞춤)
        # 시트가 하나면 파일 이름 그대로, 여러 개면 '파일이름__시트이름.csv'
        stem = path.stem if len(sheets) == 1 else f"{path.stem}__{safe_name(sheet_name)}"
        out = out_dir / f"{stem}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        results.append(f"{out.name}  (시트 '{sheet_name}', {len(df):,}행 × {df.shape[1]}열)")
    return results


def convert_folder(dir_name: str) -> None:
    """data/<dir_name>/ 의 파일을 변환해 data/<dir_name>_utf8/ 에 저장한다."""
    raw_dir = DATA_DIR / dir_name
    out_dir = DATA_DIR / f"{dir_name}{OUT_SUFFIX}"
    print(f"\n=== {raw_dir.name}/ → {out_dir.name}/ ===")

    if not raw_dir.exists():
        print(f"원본 폴더가 없습니다: {raw_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)  # 결과 폴더가 없으면 만듦

    handlers = {".csv": convert_csv, ".xlsx": convert_xlsx}
    ok, failed, skipped = 0, 0, 0

    for path in sorted(raw_dir.iterdir()):
        # .gitkeep, 엑셀이 열려 있을 때 생기는 임시파일(~$로 시작) 등은 무시
        if not path.is_file() or path.name.startswith(("~$", ".")):
            continue
        handler = handlers.get(path.suffix.lower())
        if handler is None:
            print(f"[건너뜀] {path.name}  (지원하지 않는 형식: {path.suffix})")
            skipped += 1
            continue
        try:
            for line in handler(path, out_dir):  # 저장할 폴더를 인자로 넘겨줌
                print(f"[완료]  {path.name} → {line}")
            ok += 1
        except Exception as e:  # 한 파일이 실패해도 나머지는 계속 변환
            print(f"[실패]  {path.name}  ({type(e).__name__}: {e})")
            failed += 1

    print(f"\n완료 {ok}개 / 실패 {failed}개 / 건너뜀 {skipped}개 → 결과 폴더: {out_dir}")


def main() -> None:
    # sorted(...)로 폴더 목록을 '먼저' 확정: 반복 도중 _utf8 폴더가 새로 생겨도 목록에 끼어들지 않음
    folders = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    for folder in folders:
        # 이미 변환 결과인 폴더(…_utf8)는 다시 변환하지 않음 → raw_utf8_utf8 같은 폴더가 생기는 것 방지
        if folder.name.endswith(OUT_SUFFIX):
            continue
        convert_folder(folder.name)  # Path 객체가 아니라 폴더 '이름'(문자열)을 넘김



if __name__ == "__main__":
    main()


def prepare_data(df : pd.DataFrame): pd.DataFrame
    pd.info()
"""① 날짜: base_year_month → year, month"""