import cv2
import numpy as np
import tensorflow as tf
from classification_assignment import data_labels

def extract_center_square(frame):
    h, w, _ = frame.shape
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    y2 = y1 + side
    x2 = x1 + side
    square = frame[y1:y2, x1:x2]
    return square, (x1, y1, x2, y2)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    model_path = "cifar10_cnn.model.keras"
    model = tf.keras.models.load_model(model_path)

    if cap.isOpened():
        while(cv2.waitKey(30) != ord('q')):
            ret, frame = cap.read()
            if ret == False:
                print('Video ends')
                break

            center_square, (x1, y1, x2, y2) = extract_center_square(frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            rgb_square = cv2.cvtColor(center_square, cv2.COLOR_BGR2RGB)
            resized_32 = cv2.resize(rgb_square, (32, 32), interpolation=cv2.INTER_AREA)

            video_input = resized_32.astype("float32")
            video_input = np.expand_dims(video_input, axis=0)

            probs = model.predict(video_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx]) * 100.0
            label_text = f"{data_labels[pred_idx]} ({confidence:.1f}%)"


            cv2.putText(
                frame,
                label_text,
                (x1, max(100, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (150, 255, 0),
                5,
            )

            cv2.imshow('video', frame)
    else:
        print('Video opening failed')

    cap.release()
    cv2.destroyAllWindows()
