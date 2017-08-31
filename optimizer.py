import cv2
import numpy as np
import os.path
import tempfile
from PIL import Image
from PIL import ImageFile

blur_const = 12

def blur_faces(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    result_image = img.copy()

    for (x, y, w, h) in faces:
        sub_face = img[y:y + h, x:x + w]
        sub_face = cv2.GaussianBlur(resize, (31, 31), blur_const)
        result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

    return result_image

def crop_and_blur(image, nw, nh, ow, oh):
    y_offset = int(round(oh / 4))
    h1 = int(oh - y_offset)

    crop = image[y_offset:h1, 0:ow]

    resize = cv2.resize(crop, tuple([nw, nh]), 0, 0, cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(resize, (31, 31), blur_const)

    x_offset = abs(int(round((blur.shape[1] / 2) - (image.shape[1] / 2))))
    y_offset = abs(int(round((nh - oh) / 2)))
    blur[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    return blur

def optimize_file(file_path, save_to_file, blur_face=False, buffer=False):
    if not buffer and not os.path.exists(file_path):
        return False

    tmp = tempfile.NamedTemporaryFile(delete=True,
                                    dir=os.path.dirname(os.path.abspath(__file__)),
                                    prefix="opencv-optimizer-tmp-",
                                    suffix=".jpg")

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    im = Image.open(file_path).convert("RGB")
    im.save(tmp.name, "jpeg")
    image = cv2.imread(tmp.name, cv2.IMREAD_UNCHANGED)

    need_w = 2560
    need_h = 1440
    ow = np.size(image, 1)
    oh = np.size(image, 0)
    nw = ow
    nh = oh

    if ow > need_w:
        nw = round((ow * need_h) / oh)
        nh = need_h
    elif oh > need_h:
        nh = round((oh * need_w) / ow)
        nw = need_w

    if nh != oh and nw != ow:
        image = cv2.resize(image, tuple([int(nw), int(nh)]), 0, 0, cv2.INTER_CUBIC)

        new_width = round(nh * 16 / 9)
        new_height = nh

        if new_width < nw:
            new_width = nw
            new_height = round(nw * 9 / 16)

    if blur_face:
        image = blur_faces(image)
    
    image = crop_and_blur(image, int(round(new_width)), int(round(new_height)), int(round(nw)), int(round(nh)))

    cv2.imwrite(save_to_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100,
                                      int(cv2.IMWRITE_JPEG_PROGRESSIVE), True,
                                      int(cv2.IMWRITE_JPEG_OPTIMIZE), True])
    return True
