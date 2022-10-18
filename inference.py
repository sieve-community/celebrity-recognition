import cv2

from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer    


def run(arr):

    model_labels = Labels(resources_path='resources')
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path='resources',
        use_cuda=False
    )

    o = face_recognizer.perform([cv2.resize(arr, (224, 224))])
    print(o)
    print(o[0][0][0])
    o = o[0][0][0]
    return (str(o[0]), float(o[1]))

a = cv2.imread('/Users/Mokshith/Desktop/le.png')
print(run(a))