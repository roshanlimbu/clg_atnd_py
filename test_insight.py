import cv2
import numpy as np
from insightface.app import FaceAnalysis

print("Initializing FaceAnalysis...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print("Keys in models:", app.models.keys())
rec_model = app.models.get('recognition')
print("Recognition model:", rec_model)

if rec_model:
    dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
    try:
        emb = rec_model.get_feat(dummy_img)
        print("Embedding shape via get_feat:", np.array(emb).shape)
    except Exception as e:
        print("get_feat error:", e)
