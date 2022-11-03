from typing import List, Dict
from sieve.types import Object, StaticObject, FrameFetcher
from sieve.predictors import ObjectPredictor

import cv2

from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer    
import requests

class CelebrityPredictor(ObjectPredictor):
    def setup(self):
        #Download the model
        url = "https://storage.googleapis.com/sieve-public-model-assets/celebrity_recognition/best_model_states.pkl"
        res = requests.get(url, stream = True)
        with open('resources/face_recognition/best_model_states.pkl', 'wb') as f:
            for chunk in res.iter_content(chunk_size = 1024*1024):
                if chunk:
                    f.write(chunk)

        #Load model
        model_labels = Labels(resources_path='resources')
        self.model = FaceRecognizer(
            labels=model_labels,
            resources_path='resources',
            use_cuda=False
        )
    
    def predict(self, frame_fetcher: FrameFetcher, object: Object) -> StaticObject:
        # Get bounding box from object middle frame
        object_start_frame, object_end_frame = object.start_frame, object.end_frame
        object_temporal = object.get_temporal((object_start_frame + object_end_frame)//2)
        object_bbox = object_temporal.bounding_box
        # Get image from middle frame
        frame_data = frame_fetcher.get_frame((object_start_frame + object_end_frame)//2)
        # Crop frame data to bounding box
        width = object_bbox.x2 - object_bbox.x1
        height = object_bbox.y2 - object_bbox.y1
        margin = 0.1
        margin_x1 = max(0, int(object_bbox.x1 - (margin * width)))
        margin_y1 = max(0, int(object_bbox.y1 - (margin * height)))
        margin_x2 = min(int(frame_data.shape[1]), int(object_bbox.x2 + (margin * width)))
        margin_y2 = min(int(frame_data.shape[0]), int(object_bbox.y2 + (margin * height)))
        frame_data = frame_data[margin_y1:margin_y2, margin_x1:margin_x2]

        if frame_data.shape[0] == 0 or frame_data.shape[1] == 0:
            ret_data = {"celebrity": "unknown", "celebrity_confidence": 0, "raw_celebrity": "unknown"}
            return self.get_return_val(object, **ret_data)
        
        out = self.model.perform([cv2.resize(frame_data, (224, 224))])
        if float(out[0][0][0][1]) < 0.5:
            ret_data = {"celebrity": "unknown", "celebrity_confidence": float(out[0][0][0][1]), "raw_celebrity": str(out[0][0][0][0])}
        else:
            ret_data = {"celebrity": str(out[0][0][0][0]), "celebrity_confidence": float(out[0][0][0][1]), "raw_celebrity": str(out[0][0][0][0])}
        return self.get_return_val(object, **ret_data)

    # Helper method
    def get_return_val(self, object: Object, **data) -> StaticObject:
        return StaticObject(cls=object.cls, object_id = object.object_id, start_frame=object.start_frame, end_frame=object.end_frame, skip_frames=object.skip_frames, **data)


    