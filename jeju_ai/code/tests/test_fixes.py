"""
2026-11-16 수정 3가지를 LLM 없이 확인하는 테스트 (API 키 없이 돌아감, 호출 비용 0)

실행 (code 폴더에서):  python -m pytest -q

  1. 설정: API 키·모델을 '화면 입력 → .env → 환경변수 → 기본값' 순서로 고르는가
  2. 인사·잡담: LLM을 부르지 않고 답하는가, 진짜 질문은 놓치지 않는가
  3. 표: 하위 폴더의 CSV도 읽는가, 표 보관소가 비워져도 다시 읽는가, 이름이 살짝 달라도 찾는가
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code 폴더를 import 경로에 추가

import agent  # noqa: E402
import tools  # noqa: E402

CSV = "base_year_month,biz_nm,use_amount\n202401,면세점,100\n202402,한식 음식점업,200\n"


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """테스트마다 표 보관소·설정을 깨끗한 상태로 만든다. (진짜 .env와 data 폴더는 건드리지 않음)

    [학습 포인트] tmp_path = pytest가 테스트마다 새로 만들어 주는 빈 임시 폴더.
      monkeypatch.setattr = 테스트 동안만 값을 바꿔 끼우고, 끝나면 자동으로 원래대로 돌려놓는다.
    """
    tools.TABLES.clear()
    tools.RESULTS.clear()
    monkeypatch.setattr(tools, "_builtin_results", None)
    monkeypatch.setattr(tools, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(agent, "ENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    yield
    tools.TABLES.clear()


def write_env(text: str, encoding: str = "utf-8") -> None:
    agent.ENV_PATH.write_text(text, encoding=encoding)


# ---------------------------------------------------------------------------
# 1. 설정 (API 키·모델 기본값)
# ---------------------------------------------------------------------------
def test_no_env_file_gives_fallback_model_and_no_key():
    s = agent.resolve_settings()
    assert s["api_key"] == "" and s["key_source"] is None and s["env_file"] is False
    assert s["model"] == agent.FALLBACK_MODEL and s["model_source"] == "기본값"


def test_env_file_is_used_and_edit_is_reflected_without_restart():
    write_env("GEMINI_API_KEY=AIza-first\nGEMINI_MODEL=model-a\n")
    s = agent.resolve_settings()
    assert (s["api_key"], s["key_source"], s["model"]) == ("AIza-first", ".env 파일", "model-a")
    write_env("GEMINI_API_KEY=AIza-second\n")  # 앱을 켜 둔 채 .env를 고친 상황
    s = agent.resolve_settings()
    assert s["api_key"] == "AIza-second" and s["model"] == agent.FALLBACK_MODEL


def test_screen_input_wins_over_env_file():
    write_env("GEMINI_API_KEY=AIza-env\nGEMINI_MODEL=model-env\n")
    s = agent.resolve_settings(api_key="  AIza-screen \n", model="model-screen")
    assert (s["api_key"], s["key_source"]) == ("AIza-screen", "화면 입력")  # 앞뒤 공백·줄바꿈은 떼어 냄
    assert (s["model"], s["model_source"]) == ("model-screen", "화면 입력")


def test_placeholder_key_is_treated_as_missing():
    write_env("GEMINI_API_KEY=여기에_키_붙여넣기\n")
    s = agent.resolve_settings()
    assert s["api_key"] == "" and ".env 파일" in s["key_problem"]
    out = agent.run_agent("2024년 합계 알려 줘")
    assert out["error"] == "no_key" and out["usage"]["calls"] == 0


def test_env_file_saved_by_notepad_is_readable():
    write_env("GEMINI_API_KEY=AIza-bom\n", encoding="utf-8-sig")  # 맨 앞에 BOM이 붙은 파일
    assert agent.resolve_settings()["api_key"] == "AIza-bom"
    write_env("# 한글 주석\nGEMINI_API_KEY=AIza-ansi\n", encoding="cp949")  # 'ANSI'로 저장한 파일
    assert agent.resolve_settings()["api_key"] == "AIza-ansi"


def test_status_never_contains_the_key():
    write_env("GEMINI_API_KEY=AIza-secret\n")
    status = agent.settings_status()
    assert status["has_key"] is True and "AIza-secret" not in str(status)


# ---------------------------------------------------------------------------
# 2. 인사·잡담
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, kind", [
    ("안녕", "인사"), ("안녕하세요~!", "인사"), ("안녕 ㅎㅎ", "인사"), ("Hi", "인사"), ("안녕하세요 반갑습니다", "인사"),
    ("고마워", "감사"), ("감사합니다!!", "감사"), ("안녕히 계세요", "작별"),
    ("넌 누구야?", "소개"), ("뭐 할 수 있어?", "소개"), ("도움말", "소개"),
    ("ㅋㅋㅋ", "빈말"), ("...", "빈말"), ("😀", "빈말"),
])
def test_small_talk_is_detected(text, kind):
    assert agent.classify_small_talk(text) == kind


@pytest.mark.parametrize("text", [
    "안녕, 2024년 업종별 이용 금액 알려 줘",  # 인사 + 질문 → 질문을 놓치면 안 됨
    "고마워 그럼 2023년은?",
    "면세점 이용 금액 합계는?",
    "어떤 데이터가 있어?",
    "하이브리드 차량 등록 대수",  # '하이'로 시작하지만 인사가 아님
    "销售额最高的行业是什么?",  # 한글·영문이 아닌 글자로 쓴 질문 → 빈말로 버리지 말고 LLM에게 넘겨야 함
    "売上が一番多い業種は？",
])
def test_real_questions_are_not_small_talk(text):
    assert agent.classify_small_talk(text) is None


def test_greeting_needs_no_key_and_no_llm_call(monkeypatch):
    tools.register_table("fake", CSV.encode(), title="가짜 카드 데이터")
    monkeypatch.setattr(agent, "call_llm", lambda *a, **k: pytest.fail("LLM을 부르면 안 됨"))
    out = agent.run_agent("안녕")
    assert out["error"] is None and out["usage"]["calls"] == 0 and out["steps"] == []
    assert "가짜 카드 데이터" in out["answer"]  # 지금 있는 데이터를 소개함


def test_greeting_without_any_table():
    assert "CSV를 올려" in agent.run_agent("안녕하세요")["answer"]


# ---------------------------------------------------------------------------
# 3. 표 불러오기·찾기
# ---------------------------------------------------------------------------
def make_data_folder():
    d = tools.DATA_DIR
    (d / "hub_utf8").mkdir(parents=True)
    (d / "raw").mkdir()
    (d / "card_usage.csv").write_text(CSV, encoding="utf-8")
    (d / "hub_utf8" / "제주시 유동인구(2024).csv").write_text(CSV, encoding="utf-8-sig")
    (d / "hub_utf8" / "card_usage.csv").write_text(CSV, encoding="utf-8")  # 같은 이름 → 한 번만 등록
    (d / "raw" / "원본.csv").write_text(CSV, encoding="cp949")  # 원본 폴더 → 읽지 않음
    (d / "data_dictionary.csv").write_text("a,b\n1,2\n", encoding="utf-8")  # 사전 파일 → 표가 아님
    (d / "BIG.CSV").write_text(CSV, encoding="utf-8")  # 확장자 대문자


def test_subfolder_csv_is_loaded_and_names_are_safe():
    make_data_folder()
    results = tools.ensure_builtin_tables()
    assert not [r for r in results if "error" in r]
    assert set(tools.TABLES) == {"BIG", "card_usage", "제주시_유동인구_2024"}
    assert tools.TABLES["제주시_유동인구_2024"]["title"] == "제주시 유동인구(2024)"  # 제목은 원래 파일 이름
    assert tools.TABLES["card_usage"]["title"].startswith("제주 업종별")  # BUILTIN_INFO의 제목


def test_duplicate_copy_is_skipped_but_reported():
    make_data_folder()
    results = tools.ensure_builtin_tables()
    skipped = [r for r in results if "skipped" in r]
    assert len(skipped) == 1 and skipped[0]["file"].endswith("card_usage.csv")  # hub_utf8 안의 같은 이름 파일
    assert "card_usage.csv" in skipped[0]["skipped"]  # 어느 파일 때문에 건너뛰었는지 적혀 있음


def test_different_files_with_colliding_names_are_all_loaded():
    """파일 이름은 다른데 표 이름이 같아지는 경우 → 하나도 빠지지 않고 전부 등록돼야 한다."""
    d = tools.DATA_DIR
    d.mkdir(parents=True)
    long_name = "제주특별자치도_제주시_읍면동별_업종별_성별_연령대별_월별_카드이용현황"  # 40자를 넘는 공통 앞부분
    files = [f"{long_name}_2023.csv", f"{long_name}_2024.csv", "방문객(2023).csv", "방문객_2023.csv"]
    for f in files:
        (d / f).write_text(CSV, encoding="utf-8")
    results = tools.ensure_builtin_tables()
    assert len(tools.TABLES) == 4 and not [r for r in results if "error" in r or "skipped" in r]
    assert {t["title"] for t in tools.TABLES.values()} == {Path(f).stem for f in files}  # 제목으로 서로 구분됨
    assert all(len(name) <= 60 for name in tools.TABLES)
    tools.ensure_builtin_tables(rescan=True)  # 다시 읽어도 늘어나지 않음
    assert len(tools.TABLES) == 4


def test_second_visitor_arriving_during_loading_does_not_duplicate(monkeypatch):
    """첫 접속이 큰 파일을 읽는 도중에 두 번째 접속이 들어와도 같은 파일이 두 번 등록되지 않아야 한다.

    잠금(_load_lock)이 없으면: 두 번째 접속이 '아직 아무것도 안 읽힌 상태'를 보고 따로 읽기 시작하고,
    그사이 첫 접속이 등록을 끝낸 표와 이름이 겹쳐 card_usage_2 같은 중복 표가 생긴다.
    """
    import threading
    import time
    make_data_folder()
    real = tools.register_table

    def slow_first_file(name, source, **kwargs):  # 첫 파일(BIG.CSV)만 읽는 데 오래 걸리는 상황을 흉내
        if Path(source).name == "BIG.CSV":
            time.sleep(0.2)
        return real(name, source, **kwargs)

    monkeypatch.setattr(tools, "register_table", slow_first_file)
    first = threading.Thread(target=tools.ensure_builtin_tables)
    second = threading.Thread(target=tools.ensure_builtin_tables)
    first.start()
    time.sleep(0.1)  # 첫 접속이 BIG.CSV를 읽는 중일 때
    second.start()
    first.join()
    second.join()
    assert sorted(tools.TABLES) == ["BIG", "card_usage", "제주시_유동인구_2024"]


def test_tables_come_back_after_store_is_emptied(monkeypatch):
    """Streamlit이 tools 모듈을 다시 import한 상황 = TABLES가 비고, 읽음 표시도 None으로 돌아간 상태."""
    make_data_folder()
    tools.ensure_builtin_tables()
    tools.TABLES.clear()
    monkeypatch.setattr(tools, "_builtin_results", None)
    tools.ensure_builtin_tables()
    assert len(tools.TABLES) == 3


def test_rescan_picks_up_new_file_only():
    make_data_folder()
    tools.ensure_builtin_tables()
    first = tools.TABLES["card_usage"]["df"]
    (tools.DATA_DIR / "new_data.csv").write_text(CSV, encoding="utf-8")
    assert "new_data" not in tools.TABLES and tools.ensure_builtin_tables() and "new_data" not in tools.TABLES
    tools.ensure_builtin_tables(rescan=True)
    assert "new_data" in tools.TABLES
    assert tools.TABLES["card_usage"]["df"] is first  # 이미 있던 표는 다시 읽지 않음


def test_broken_file_error_names_the_file():
    tools.DATA_DIR.mkdir(parents=True)
    (tools.DATA_DIR / "empty.csv").write_text("a,b\n", encoding="utf-8")  # 머리글만 있고 행이 없음
    results = tools.ensure_builtin_tables()
    assert "empty.csv" in results[0]["error"]


def test_table_is_found_even_if_name_differs_slightly():
    make_data_folder()
    tools.ensure_builtin_tables()
    for written in ["제주시_유동인구_2024", "제주시 유동인구(2024)", "제주시 유동인구(2024).csv", " CARD_USAGE "]:
        assert "error" not in tools.describe_data(written), written
    assert tools.describe_data("제주시 유동인구(2024)")["name"] == "제주시_유동인구_2024"  # 진짜 이름을 돌려줌
    err = tools.describe_data("유동인구")["error"]  # 비슷하기만 한 이름은 추측하지 않고 목록을 알려 줌
    assert "제주시_유동인구_2024" in err and "card_usage" in err


def test_prompt_lists_tables_and_focus():
    make_data_folder()
    tools.ensure_builtin_tables()
    prompt = agent.build_system_prompt(focus_table="card_usage")
    assert "- 제주시_유동인구_2024: 제주시 유동인구(2024) / 2행, 기간 2024-01 ~ 2024-02" in prompt
    assert "지금 화면에서 보고 있는 표: card_usage" in prompt
    assert "화면에서 보고 있는 표" not in agent.build_system_prompt(focus_table="없는표")