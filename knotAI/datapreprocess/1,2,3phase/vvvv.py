"""
랜덤 간격 자동 클릭 매크로 (Windows 전용)
  좌클릭 1회 -> 잠깐 대기 -> Enter 1회  를 한 사이클로 반복
  사이클 간격: 1.0 ~ 2.0초 랜덤

  F8 : 시작 / 일시정지
  F9 : 종료
"""
import ctypes
import ctypes.wintypes as wt
import random
import threading
import time

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE    = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004
KEYEVENTF_KEYUP      = 0x0002
KEYEVENTF_SCANCODE   = 0x0008
SC_ENTER   = 0x1C
VK_RETURN  = 0x0D
VK_F8, VK_F9 = 0x77, 0x78

# ── 설정 ──────────────────────────────────────
CLICK_INTERVAL = (1.0, 2.0)    # 사이클 간격(초) 랜덤 범위
ENTER_DELAY    = (0.05, 0.15)  # 클릭 후 Enter 까지 딜레이
CLICK_HOLD     = (0.03, 0.08)  # 버튼 누르고 있는 시간
USE_SCANCODE   = True          # 게임에서 안 먹으면 False
# ─────────────────────────────────────────────

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]


user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wt.UINT


def _send(*inputs):
    arr = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        print("[경고] SendInput 실패 (전송:%d, err=%d)" % (sent, ctypes.get_last_error()))


def click():
    """현재 커서 위치에서 좌클릭 1회 (누름/뗌 사이에 짧은 홀드)"""
    _send(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0)))
    time.sleep(random.uniform(*CLICK_HOLD))
    _send(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0)))


def press_enter():
    if USE_SCANCODE:
        flags, vk, sc = KEYEVENTF_SCANCODE, 0, SC_ENTER
    else:
        flags, vk, sc = 0, VK_RETURN, 0
    _send(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, sc, flags, 0, 0)))
    time.sleep(random.uniform(0.03, 0.07))
    _send(INPUT(type=INPUT_KEYBOARD,
                ki=KEYBDINPUT(vk, sc, flags | KEYEVENTF_KEYUP, 0, 0)))


running = threading.Event()   # 동작 중 여부
stopping = threading.Event()  # 종료 신호


def worker():
    count = 0
    while not stopping.is_set():
        if not running.wait(timeout=0.1):
            continue
        click()
        time.sleep(random.uniform(*ENTER_DELAY))
        press_enter()
        count += 1

        wait = random.uniform(*CLICK_INTERVAL)
        print("[%d회] 다음까지 %.2f초" % (count, wait))
        # 대기 중에도 F8/F9 반응하도록 잘게 나눠서 sleep
        end = time.time() + wait
        while time.time() < end and running.is_set() and not stopping.is_set():
            time.sleep(0.02)


def main():
    expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
    print("sizeof(INPUT) = %d (기대값 %d)" % (ctypes.sizeof(INPUT), expected))
    print("F8: 시작/일시정지   F9: 종료")

    threading.Thread(target=worker, daemon=True).start()

    f8_prev = f9_prev = False
    while True:
        f8 = bool(user32.GetAsyncKeyState(VK_F8) & 0x8000)
        f9 = bool(user32.GetAsyncKeyState(VK_F9) & 0x8000)
        if f8 and not f8_prev:
            if running.is_set():
                running.clear()
                print(">>> 일시정지")
            else:
                running.set()
                print(">>> 시작")
        if f9 and not f9_prev:
            stopping.set()
            running.set()      # worker 의 wait 깨우기
            print(">>> 종료")
            break
        f8_prev, f9_prev = f8, f9
        time.sleep(0.03)

    time.sleep(0.2)


if __name__ == "__main__":
    main()
