import face_recognition
import dlib
from facenet_pytorch import MTCNN
import cv2
from matplotlib import pyplot as plt
from .video_tools import read_video_decord
dlib.DLIB_USE_CUDA = True
face_detector = dlib.get_frontal_face_detector()
faceCascade = cv2.CascadeClassifier(
            r"D:\These\Git\Audio_Visual\src\mdvae\tools\haarcascade_frontalface_default.xml")
mtcnn = MTCNN(device='cuda', image_size=(720, 1280))


def face_detection_mctnn(frame, resize=None):
    bbox = mtcnn.detect(frame)
    if bbox[0] is not None:
        bbox = bbox[0][0]
        bbox = [round(x) for x in bbox]
        x1, y1, x2, y2 = bbox
    cropped_image = frame[y1-40:y2+40, x1-40:x2+40, :]
    if resize is not None:
        cropped_image = cv2.resize(cropped_image, dsize=resize, interpolation=cv2.INTER_CUBIC)
    return cropped_image


def face_detection_cascade(image, resize=None):
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=20, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cropped_image = image[y-30:y + h + 30, x-30:x + w+30]
    if resize is not None:
        cropped_image = cv2.resize(cropped_image, dsize=resize, interpolation=cv2.INTER_CUBIC)
    return cropped_image


def face_detection_(frame, resize=None):
    faces = face_detector(frame, 1)
    for face in faces:
        cropped_image = frame[face.top()-30:face.bottom() + 20, face.left() - 20:face.right() + 20]
    if resize is not None:
        cropped_image = cv2.resize(cropped_image, dsize=resize, interpolation=cv2.INTER_CUBIC)
        # cropped_image = frame[face.top():face.bottom(), face.left():face.right()]
    return cropped_image


if __name__ == '__main__':
    images = read_video_decord(file_path=r"D:\These\data\Audio-Visual\RAVDESS\Ravdess-visual\Actor_01\01-01-02-01-01-01-01.mp4")
    print(images.shape)
    # print(images.shape)
    # for i, frame in enumerate(images):
    #     cropped_image = face_detection(frame, resize=None)
    #     print(i)
    #     plt.imshow(cropped_image)
    #     # plt.imshow(cropped_image)
    #     plt.savefig(f'temps/images_{i}')
