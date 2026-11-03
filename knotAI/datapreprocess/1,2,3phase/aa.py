import os
import cv2
import numpy as np

# -------------------------------------------------------------------------
# 1. 한글 경로 버전과 영문(My Drive) 경로 버전 준비
# -------------------------------------------------------------------------
img_korean = r"G:\내 드라이브\knot_dataset\10knots\knot_labeling_image_data\knot_image\20260725_004503_jpg.rf.niEUJjOJEc9dU1uuGMAH.jpg"
img_english = r"G:\My Drive\knot_dataset\10knots\knot_labeling_image_data\knot_image\20260725_004503_jpg.rf.niEUJjOJEc9dU1uuGMAH.jpg"

lbl_korean = r"G:\내 드라이브\knot_dataset\10knots\knot_labeling_image_data\knot_label_data\20260725_004503_jpg.rf.niEUJjOJEc9dU1uuGMAH.txt"
lbl_english = r"G:\My Drive\knot_dataset\10knots\knot_labeling_image_data\knot_label_data\20260725_004503_jpg.rf.niEUJjOJEc9dU1uuGMAH.txt"

print("🔍 [1단계] 파일 경로 존재 여부 테스트 중...\n")

real_image_path = None
real_label_path = None

# 이미지 경로 테스트
if os.path.exists(img_korean):
    print("✅ 이미지 파일 발견! (한글 '내 드라이브' 경로 인식 성공)")
    real_image_path = img_korean
elif os.path.exists(img_english):
    print("✅ 이미지 파일 발견! (영문 'My Drive' 경로로 인식 성공)")
    real_image_path = img_english
else:
    print("❌ 이미지 파일을 찾을 수 없습니다. (파일명 오타 또는 폴더 위치 확인 필요)")

# 라벨 경로 테스트
if os.path.exists(lbl_korean):
    print("✅ 라벨 txt 파일 발견! (한글 '내 드라이브' 경로 인식 성공)")
    real_label_path = lbl_korean
elif os.path.exists(lbl_english):
    print("✅ 라벨 txt 파일 발견! (영문 'My Drive' 경로로 인식 성공)")
    real_label_path = lbl_english
else:
    print("❌ 라벨 txt 파일을 찾을 수 없습니다.")


# -------------------------------------------------------------------------
# 2. [OpenCV 한글 경로 대응] 이미지 및 라벨 안전하게 읽기
# -------------------------------------------------------------------------
if real_image_path and real_label_path:
    print("\n🚀 [2단계] 이미지 및 라벨 데이터 읽기 시도 중...")
    
    # 1) OpenCV 한글 경로 읽기 (np.fromfile 필수)
    img_array = np.fromfile(real_image_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 2) 라벨 txt 파일 읽기
    with open(real_label_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
        
    print("\n" + "="*50)
    print(f"🎉 성공! 이미지를 정상적으로 읽었습니다. (크기: {image.shape})")
    print(f"📄 라벨 내용 ({len(labels)}개 객체):")
    for l in labels:
        print("   ", l.strip())
    print("="*50)
    
    # 올바른 경로를 변수에 최종 저장
    image_path = real_image_path
    label_path = real_label_path
else:
    print("\n💡 [팁] 윈도우 탐색기에서 해당 이미지 파일 클릭 후 'Shift + 마우스 우클릭' -> [경로로 복사]를 눌러 주소를 다시 복사해 보세요!")
    