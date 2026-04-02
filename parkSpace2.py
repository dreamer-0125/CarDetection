import cv2
import cvzone
import numpy as np
import pyzbar.pyzbar as pyzbar

cap = cv2.VideoCapture("park3.webm")
cam = cv2.VideoCapture(0)

with open("CarPos", "rb") as f:
    posList = pickle.load(f)

posList.sort(key=lambda item: item[1])

width, height = 42, 92

# Motion: blurred grayscale diff + threshold (more stable than Canny edge diff)
GAUSSIAN_BLUR = (5, 5)
DIFF_THRESH = 22  # min |I - I_prev| to count as changed (raise if shadows/noise)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Fraction of ROI pixels that must be "hot" to flag motion (scales with slot size)
MOTION_FRACTION = 0.028
MOTION_PIXEL_MIN = 96  # floor so tiny ROIs are not overly sensitive


def preprocess(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, GAUSSIAN_BLUR, 0)


def motion_mask(prev_blur, curr_blur):
    diff = cv2.absdiff(curr_blur, prev_blur)
    _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)
    return mask


def detectActivity(display_bgr, mask):
    global move_hist

    roi_area = width * height
    motion_threshold = max(int(MOTION_FRACTION * roi_area), MOTION_PIXEL_MIN)

    space_id = 0
    suspicious = set()

    for pos in posList:
        space_id += 1
        x, y = pos
        roi = mask[y : y + height, x : x + width]
        count = cv2.countNonZero(roi)
        sid = str(space_id)

        if count < motion_threshold:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            move_hist.add(sid)
            if sid not in allowed:
                suspicious.add(sid)

        cv2.rectangle(
            display_bgr,
            pos,
            (pos[0] + width, pos[1] + height),
            color,
            2,
        )
        cvzone.putTextRect(
            display_bgr,
            " id: " + sid,
            (x, y + height - 3),
            scale=0.75,
            thickness=1,
            offset=0,
            colorR=(0, 0, 0),
        )

    if suspicious:
        cvzone.putTextRect(
            display_bgr,
            f'Possible theft at space ID {", ".join(sorted(suspicious, key=int))}',
            (25, 50),
            scale=2,
            thickness=2,
            colorR=(0, 0, 200),
        )


img0 = cv2.imread("first_frame.jpg")
prev_blur = preprocess(img0)
reference_blur = prev_blur.copy()

move_hist = set()
allowed = set()
last_identity = ""
frame_count = 0
last_frame_count = -50

while True:
    frame_count += 1
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        print("-------------------start-------------------")
        move_hist = set()
        prev_blur = reference_blur.copy()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        last_frame_count = -50

    ok_cam, cam_img = cam.read()
    if not ok_cam:
        cam_img = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        cam_img = cv2.resize(
            cam_img,
            (int(cam_img.shape[1] * 0.5), int(cam_img.shape[0] * 0.5)),
            interpolation=cv2.INTER_CUBIC,
        )

    if (frame_count - last_frame_count) > 50:
        for obj in pyzbar.decode(cam_img):
            last_frame_count = frame_count
            raw = obj.data
            identity = (
                raw.decode("utf-8", errors="replace")
                if isinstance(raw, bytes)
                else str(raw)
            )
            last_identity = identity
            if identity not in allowed:
                allowed.add(identity)
            else:
                allowed.remove(identity)
    elif last_identity:
        cvzone.putTextRect(
            cam_img,
            f"Last scan / toggled id: {last_identity}",
            (25, 50),
            scale=1,
            thickness=1,
            colorR=(0, 200, 0),
        )

    cv2.imshow("image", cam_img)

    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(
        img,
        (int(img.shape[1] * 1.5), int(img.shape[0] * 1.5)),
        interpolation=cv2.INTER_CUBIC,
    )

    curr_blur = preprocess(img)
    mask = motion_mask(prev_blur, curr_blur)
    detectActivity(img, mask)
    prev_blur = curr_blur.copy()

    cv2.imshow("Image", img)

    if cv2.waitKey(5) & 0xFF == ord("d"):
        break
