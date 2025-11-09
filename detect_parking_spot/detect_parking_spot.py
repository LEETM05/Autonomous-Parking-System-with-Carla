from ultralytics import YOLO
import numpy as np
import json
import math
import argparse
import cv2 # 💡 추가
from PIL import Image, ImageDraw, ImageFont # 💡 추가

# [수정 전] convert_obb_to_four_corners 함수 (YOLO OBB는 이미 네 꼭짓점을 반환하므로 필요 없어짐)

def draw_obb(image_np, obb_corners, text, color=(0, 0, 255), thickness=2):
    """
    numpy 배열 이미지에 회전된 바운딩 박스를 그립니다.
    obb_corners는 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 형식입니다.
    """
    points = np.array(obb_corners, dtype=np.int32)
    cv2.polylines(image_np, [points], isClosed=True, color=color, thickness=thickness)

    # 텍스트 추가 (PIL을 사용하여 한글 지원 및 깨짐 방지)
    img_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 폰트 설정 (시스템에 한글 폰트가 설치되어 있어야 합니다. 예: NotoSansKR)
    try:
        font = ImageFont.truetype("NotoSansKR-Regular.otf", 18) # 💡 폰트 경로와 크기 조정
    except IOError:
        font = ImageFont.load_default() # 폰트가 없으면 기본 폰트 사용

    # 박스의 첫 번째 꼭짓점 근처에 텍스트를 배치
    text_pos = (int(obb_corners[0][0]), int(obb_corners[0][1] - 25)) # y축으로 25픽셀 위로 이동
    
    # 텍스트 외곽선 (선명하게 보이도록)
    for x_offset in [-1, 1]:
        for y_offset in [-1, 1]:
            draw.text((text_pos[0] + x_offset, text_pos[1] + y_offset), text, font=font, fill=(0,0,0)) # 검은색 외곽선
    draw.text(text_pos, text, font=font, fill=color) # 박스 색깔로 본문 텍스트

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main(image_path, model_path, output_json_path, visualize_output_path=None):
    # 학습된 모델 로드
    model = YOLO(model_path)
    
    # 이미지에서 객체 탐지
    results = model(image_path)
    
    detected_spots = []
    
    # 💡 [추가] 시각화용 이미지 로드
    img_for_vis = cv2.imread(image_path)
    if img_for_vis is None:
        print(f"Detector: Error loading image for visualization: {image_path}")
        return

    for r in results:
        # OBB 결과가 있는지 확인
        if r.obb is not None:
            # 원본 이미지 크기 (정규화된 좌표를 픽셀로 변환하기 위함)
            # YOLOv8의 r.obb.xyxyxyxy는 이미 픽셀 좌표이므로 변환 불필요
            img_width, img_height = r.orig_shape[1], r.orig_shape[0]

            for i, obb in enumerate(r.obb):
                # xyxyxyxy: [x1, y1, x2, y2, x3, y3, x4, y4]
                # 이 좌표는 이미 픽셀 단위입니다.
                # numpy 배열로 변환 후 reshape하여 [[x1,y1], [x2,y2], ... ] 형태로 만듭니다.
                obb_corners_flat = obb.xyxyxyxy.cpu().numpy()[0]
                obb_corners = obb_corners_flat.reshape(-1, 2).tolist() # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                
                confidence = float(obb.conf.cpu().numpy()[0])
                class_id = int(obb.cls.cpu().numpy()[0])
                
                # 클래스 이름 매핑 (data.yaml의 names와 일치시켜야 합니다)
                class_name = model.names[class_id] # YOLO 모델 내부에 names 정보가 있습니다.

                detected_spots.append({
                    'confidence': confidence,
                    'corners': obb_corners, # 이미 픽셀 좌표
                    'class_name': class_name
                })
                
                # 💡 [추가] 시각화: 바운딩 박스와 신뢰도 점수를 이미지에 그립니다.
                text = f"{class_name} {confidence:.1f}"
                img_for_vis = draw_obb(img_for_vis, obb_corners, text)

    # 가장 신뢰도 높은 순으로 정렬
    detected_spots.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 결과를 JSON 파일로 저장
    with open(output_json_path, 'w') as f:
        json.dump(detected_spots, f, indent=4)
        
    print(f"Detector: Found {len(detected_spots)} spots. Results saved to {output_json_path}")

    # 💡 [추가] 시각화된 이미지 저장
    if visualize_output_path:
        cv2.imwrite(visualize_output_path, img_for_vis)
        print(f"Detector: Visualized image saved to {visualize_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to the trained YOLO model.')
    parser.add_argument('--output_json', type=str, default='result.json', help='Path to the output JSON file.')
    # 💡 [추가] 시각화된 이미지를 저장할 경로 인자
    parser.add_argument('--output_vis_img', type=str, default='_detected_parking_spot.png', 
                        help='Path to save the visualized image.')
    args = parser.parse_args()
    
    main(args.image, args.model, args.output_json, args.output_vis_img)