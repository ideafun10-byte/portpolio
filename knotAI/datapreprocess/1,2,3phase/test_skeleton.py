import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

# ==========================================
# 1. 폴더 경로 설정 (이미지 폴더 및 라벨 폴더)
# ==========================================
image_dir = r"G:\내 드라이브\knot_dataset\10knots\knot_labeling_image_data\knot_image"
label_dir = r"G:\내 드라이브\knot_dataset\10knots\knot_labeling_image_data\knot_label_data"


def run_skeletonization_for_file(img_p, lbl_p, filename):
    # 2. 원본 이미지 로드 (한글 경로 대응 np.fromfile)
    try:
        img_array = np.fromfile(img_p, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ 이미지 로드 실패 ({filename}): {e}")
        return

    if img is None:
        print(f"❌ 이미지를 읽을 수 없습니다 ({filename}).")
        return

    h, w, _ = img.shape

    # 3. 빈 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)

    # 4. .txt 라벨 파일 읽어서 폴리곤 그리드 복원
    if not os.path.exists(lbl_p):
        print(f"⚠️ 매칭되는 라벨 파일이 없습니다 ({filename}).")
        return

    with open(lbl_p, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        coords = [float(x) for x in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i + 1] * h)
            points.append([px, py])

        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # 5. 마스크 이진화 (0과 1)
    binary_mask = mask > 128

    # 6. 스켈레톤화 (1픽셀 두께 중심선만 남기기)
    skeleton = skeletonize(binary_mask)
    skeleton_img = (skeleton * 255).astype(np.uint8)

    # 7. 원본 이미지 위에 빨간색 스켈레톤 선 합성
    overlay_img = img.copy()
    overlay_img[skeleton] = [0, 0, 255]  # BGR 기준 빨간색

    # 8. Matplotlib 출력 (현재 띄워진 창을 닫으면 다음 이미지로 자동 이동)
    plt.figure(figsize=(16, 4))
    plt.suptitle(f"File [{filename}]", fontsize=13, fontweight="bold")

    plt.subplot(1, 4, 1)
    plt.title("1. Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("2. Reconstructed Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("3. Skeleton (1px)")
    plt.imshow(skeleton_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("4. Skeleton Overlay")
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()  # 💡 창을 닫으면 자동으로 다음 파일 처리로 넘어갑니다!


# ==========================================
# 메인 루프: 이미지 폴더 안의 모든 파일 탐색
# ==========================================
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

if not os.path.exists(image_dir):
    print("❌ 이미지 폴더 경로가 잘못되었습니다.")
else:
    # 이미지 파일 목록 탐색
    all_files = os.listdir(image_dir)
    image_files = [
        f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    print(
        f"📂 총 {len(image_files)}개의 이미지 파일을 발견했습니다. 순서대로 출력을 시작합니다.\n"
    )

    for idx, img_filename in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 시각화 처리 중: {img_filename}")

        # 확장자 제거 후 .txt 라벨 파일명 매칭
        base_name = os.path.splitext(img_filename)[0]
        lbl_filename = base_name + ".txt"

        full_img_path = os.path.join(image_dir, img_filename)
        full_lbl_path = os.path.join(label_dir, lbl_filename)

        run_skeletonization_for_file(full_img_path, full_lbl_path, img_filename)