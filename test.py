"""
    python test.py
    python test.py --model best_model_m1.keras
    python test.py --tflite eye_model_int8.tflite
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque
from pathlib import Path

IMG_SIZE       = 32
BRIGHTNESS_THR = 80       
CONF_THRESH    = 0.65
PERCLOS_WINDOW = 30
PERCLOS_THRESH = 0.35

#left eye(hafl face left)
LEFT_EYE_Y_START = 0.20
LEFT_EYE_Y_END   = 0.50
LEFT_EYE_X_START = 0.10
LEFT_EYE_X_END   = 0.45

# right eye (half face right)
RIGHT_EYE_Y_START = 0.20
RIGHT_EYE_Y_END   = 0.50
RIGHT_EYE_X_START = 0.55
RIGHT_EYE_X_END   = 0.90

#esp32 firm ware (arduino nạp code)
def simple_enhance(gray_crop: np.ndarray) -> np.ndarray:
    """
    Histogram stretch: simple_enhance() trong firmware.
    """
    mean_brightness = np.mean(gray_crop)
    if mean_brightness >= BRIGHTNESS_THR:
        return gray_crop

    min_val = np.min(gray_crop)
    max_val = np.max(gray_crop)
    if max_val == min_val:
        return gray_crop

    stretched = ((gray_crop.astype(np.float32) - min_val)
                 / (max_val - min_val) * 255).astype(np.uint8)
    return stretched


def preprocess_esp32_style(eye_crop_gray: np.ndarray,
                           use_tflite: bool = False):
    """
    Pipeline preprocess
      1. Histogram stretch nếu tối
      2. Resize → 32×32
      3. Normalize: float32 hoặc INT8
    """
    enhanced = simple_enhance(eye_crop_gray)
    resized  = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_AREA)

    if use_tflite:
        inp = (resized.astype(np.int16) - 128).astype(np.int8)
        return inp.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    else:
        inp = resized.astype(np.float32) / 255.0
        return inp.reshape(1, IMG_SIZE, IMG_SIZE, 1)


#load model
class KerasPredictor:
    def __init__(self, path):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        print(f"[INFO] Loaded Keras model: {path}")

    def predict(self, inp):
        pred = self.model.predict(inp, verbose=0)[0]
        return float(pred[0]), float(pred[1])  # p_open, p_closed


class TFLitePredictor:
    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter
        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        out_quant = self.output_details[0]['quantization']
        self.out_scale    = out_quant[0]
        self.out_zero_pt  = out_quant[1]
        print(f"[INFO] Loaded TFLite model: {path}")

    def predict(self, inp):
        self.interpreter.set_tensor(
            self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(
            self.output_details[0]['index'])[0]

        p_open   = (float(out[0]) - self.out_zero_pt) * self.out_scale
        p_closed = (float(out[1]) - self.out_zero_pt) * self.out_scale
        return p_open, p_closed


# perclos
class PerclosCounter:
    def __init__(self, window=PERCLOS_WINDOW, thresh=PERCLOS_THRESH):
        self.buffer  = deque([False] * window, maxlen=window)
        self.thresh  = thresh

    def update(self, is_closed: bool) -> tuple[float, bool]:
        self.buffer.append(is_closed)
        perclos = sum(self.buffer) / len(self.buffer)
        return perclos, perclos >= self.thresh


# demo

def get_roi_coords(fx, fy, fw, fh, w, h, x_start, x_end, y_start, y_end):
    """Hàm phụ trợ để tính tọa độ ROI và clamp không vượt quá khung hình"""
    roi_x  = int(fx + fw * x_start)
    roi_y  = int(fy + fh * y_start)
    roi_x2 = int(fx + fw * x_end)
    roi_y2 = int(fy + fh * y_end)

    roi_x  = max(0, roi_x)
    roi_y  = max(0, roi_y)
    roi_x2 = min(w, roi_x2)
    roi_y2 = min(h, roi_y2)
    return roi_x, roi_y, roi_x2, roi_y2


def run(model_path: str, tflite_path: str, cam_index: int):

    use_tflite = tflite_path is not None
    if use_tflite:
        predictor = TFLitePredictor(tflite_path)
    else:
        predictor = KerasPredictor(model_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    perclos = PerclosCounter()

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)   # giống ESP32 QVGA
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được camera {cam_index}")
        return

    print("[INFO] Running — Q: thoát | S: screenshot")

    fps_time    = time.time()
    frame_count = 0
    fps         = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (320, 240))
        h, w  = frame.shape[:2]

        frame_count += 1
        if frame_count % 20 == 0:
            fps = 20 / (time.time() - fps_time)
            fps_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        label_str   = "No face"
        perclos_val = 0.0
        drowsy      = False

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2]*r[3])
            cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (255,200,0), 1)

            #coor left and right eye roi, clamp to frame
            lx1, ly1, lx2, ly2 = get_roi_coords(fx, fy, fw, fh, w, h, 
                                                LEFT_EYE_X_START, LEFT_EYE_X_END, 
                                                LEFT_EYE_Y_START, LEFT_EYE_Y_END)
                                                
            rx1, ry1, rx2, ry2 = get_roi_coords(fx, fy, fw, fh, w, h, 
                                                RIGHT_EYE_X_START, RIGHT_EYE_X_END, 
                                                RIGHT_EYE_Y_START, RIGHT_EYE_Y_END)

            left_eye_crop = gray[ly1:ly2, lx1:lx2]
            right_eye_crop = gray[ry1:ry2, rx1:rx2]

            is_left_closed = False
            is_right_closed = False

            # left eye
            if left_eye_crop.size > 0:
                inp_left = preprocess_esp32_style(left_eye_crop, use_tflite)
                p_open_l, p_closed_l = predictor.predict(inp_left)
                conf_l = max(p_open_l, p_closed_l)
                is_left_closed = (p_closed_l > p_open_l) and (conf_l >= CONF_THRESH)
                
                color_l = (0, 0, 220) if is_left_closed else (0, 220, 0)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color_l, 2)

            #right eye
            if right_eye_crop.size > 0:
                inp_right = preprocess_esp32_style(right_eye_crop, use_tflite)
                p_open_r, p_closed_r = predictor.predict(inp_right)
                conf_r = max(p_open_r, p_closed_r)
                is_right_closed = (p_closed_r > p_open_r) and (conf_r >= CONF_THRESH)
                
                color_r = (0, 0, 220) if is_right_closed else (0, 220, 0)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color_r, 2)

            # frame = close (1 or 2 eyes closed)
            is_frame_closed = is_left_closed or is_right_closed
            label_str = f"{'CLOSED' if is_frame_closed else 'OPEN'}"
            
            perclos_val, drowsy = perclos.update(is_frame_closed)

        else:
            # face cant detect -> reset perclos
            perclos_val, drowsy = perclos.update(False)

        #status
        if drowsy:
            cv2.rectangle(frame, (0,0), (w,50), (0,0,180), -1)
            cv2.putText(frame, "DROWSY - WAKE UP!",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255,255,255), 2, cv2.LINE_AA)

        mode_str = "TFLite INT8" if use_tflite else "Keras"
        hud = [
            f"FPS: {fps:.1f}",
            f"Mode: {mode_str}",
            f"Pred: {label_str}",
            f"PERCLOS: {perclos_val:.0%} ({'DROWSY' if drowsy else 'OK'})",
        ]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (4, 18 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,255), 1, cv2.LINE_AA)

        cv2.imshow("ESP32-CAM Simulator (320x240)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="best_model_m1.keras",
                        help="Keras model (.keras)")
    parser.add_argument("--tflite", default=None,
                        help="TFLite model (.tflite)")
    parser.add_argument("--cam",    type=int, default=0)
    args = parser.parse_args()

    if args.tflite and not Path(args.tflite).exists():
        print(f"[ERROR] Không tìm thấy: {args.tflite}")
    elif not args.tflite and not Path(args.model).exists():
        print(f"[ERROR] Không tìm thấy: {args.model}")
    else:
        run(args.model, args.tflite, args.cam)