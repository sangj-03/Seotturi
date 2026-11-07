"""Find the face landmarks of a character image with OpenCV's YuNet: eyes, nose, mouth corners."""
import os
import sys, cv2, numpy as np
MODEL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "yunet.onnx")
def detect(path, thresh=0.5):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(MODEL, "", (w, h), thresh, 0.3, 5000)
    _, faces = det.detect(img)
    out = []
    for f in (faces if faces is not None else []):
        x, y, fw, fh = f[:4]
        lm = f[4:14].reshape(5, 2)  # right eye, left eye, nose, right mouth corner, left mouth corner (image left/right swapped)
        out.append({"box": [float(x), float(y), float(fw), float(fh)], "score": float(f[14]), "lm": lm.tolist()})
    return sorted(out, key=lambda f: -f["box"][2] * f["box"][3])
if __name__ == "__main__":
    for p in sys.argv[1:]:
        fs = detect(p)
        print(p.split("/")[-1], [(round(f["score"], 2), [round(v) for v in f["box"]], [[round(a) for a in pt] for pt in f["lm"]]) for f in fs][:3])
