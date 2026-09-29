"""Driver drowsiness detection using MediaPipe, EAR and MAR."""

import time, threading, winsound
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CAMERA = 0
EAR_LIMIT, MAR_LIMIT = .21, .55
EYE_TIME, YAWN_TIME = 1.2, 1.0

# Eye and mouth landmark indices.
EYES = ((33,160,158,133,153,144), (362,385,387,263,373,380))
MOUTH = (61,291,13,14)


def points(face, ids, w, h):
    return np.array([(face[i].x*w, face[i].y*h) for i in ids], np.float32)


def ear(p):
    """Calculate Eye Aspect Ratio."""
    return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (
        2*np.linalg.norm(p[0]-p[3]))


def draw_landmarks(frame, face, ids, w, h, color):
    """Draw selected facial landmarks and connecting lines."""
    p = [(int(face[i].x*w), int(face[i].y*h)) for i in ids]
    for x, y in p:
        cv2.circle(frame, (x, y), 2, color, -1)
    for a, b in zip(p, p[1:]):
        cv2.line(frame, a, b, color, 1)


def alarm(stop):
    """Continuous warning sound while drowsiness is active."""
    while not stop.is_set():
        winsound.Beep(1500, 350)
        stop.wait(.15)


def main():
    cam = cv2.VideoCapture(CAMERA)
    if not cam.isOpened():
        raise SystemExit("Webcam not available")

    opt = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)

    closed = yawn = alarm_thread = stop = None
    timestamp = 0

    with vision.FaceLandmarker.create_from_options(opt) as detector:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            image = mp.Image(
                mp.ImageFormat.SRGB,
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            timestamp += 33
            result = detector.detect_for_video(image, timestamp)

            EAR = MAR = 0
            status, alert = "NO FACE DETECTED", False

            if result.face_landmarks:
                face = result.face_landmarks[0]

                # Calculate EAR from both eyes.
                eyes = [ear(points(face, e, w, h)) for e in EYES]
                EAR = float(np.mean(eyes))

                # Calculate MAR from mouth landmarks.
                m = points(face, MOUTH, w, h)
                MAR = float(np.linalg.norm(m[2]-m[3]) /
                            max(np.linalg.norm(m[0]-m[1]), 1))

                now = time.monotonic()
                if EAR < EAR_LIMIT and closed is None: closed = now
                if MAR > MAR_LIMIT and yawn is None: yawn = now
                if EAR >= EAR_LIMIT: closed = None
                if MAR <= MAR_LIMIT: yawn = None

                eye_alert = closed and now-closed >= EYE_TIME
                yawn_alert = yawn and now-yawn >= YAWN_TIME
                alert = bool(eye_alert or yawn_alert)

                status = ("DROWSINESS ALERT" if eye_alert else
                          "YAWN DETECTED" if yawn_alert else "MONITORING")

                # Visualize the exact landmarks used for EAR/MAR.
                color = (0, 0, 255) if alert else (0, 255, 0)
                for eye in EYES:
                    draw_landmarks(frame, face, eye, w, h, color)
                draw_landmarks(frame, face, MOUTH, w, h, color)

            else:
                closed = yawn = None

            # Start/stop continuous alarm.
            if alert and alarm_thread is None:
                stop = threading.Event()
                alarm_thread = threading.Thread(
                    target=alarm, args=(stop,), daemon=True)
                alarm_thread.start()

            if not alert and alarm_thread:
                stop.set()
                alarm_thread = stop = None

            color = (0, 0, 255) if alert else (0, 255, 0)

            cv2.putText(frame, f"EAR: {EAR:.2f}  MAR: {MAR:.2f}",
                        (20, 35), 0, .65, (255,255,255), 2)
            cv2.putText(frame, f"EAR<{EAR_LIMIT}  MAR>{MAR_LIMIT}",
                        (20, 65), 0, .45, (200,200,200), 1)
            cv2.putText(frame, status, (20, 95),
                        0, .65, color, 2)

            cv2.imshow("Driver Drowsiness Monitor", frame)

            if cv2.waitKey(1) & 255 in (ord("q"), 27):
                break

        if stop:
            stop.set()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# """Driver drowsiness detection using MediaPipe EAR + MAR."""

# import time, threading, winsound
# import cv2, numpy as np, mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# CAMERA = 0
# EAR_LIMIT, MAR_LIMIT = .21, .55
# EYE_TIME, YAWN_TIME = 1.2, 1.0
# EYES = ((33,160,158,133,153,144), (362,385,387,263,373,380))
# MOUTH = (61,291,13,14)


# def points(face, ids, w, h):
#     return np.array([(face[i].x*w, face[i].y*h) for i in ids], np.float32)


# def ear(p):
#     return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (
#         2*np.linalg.norm(p[0]-p[3]))


# def alarm(stop):
#     while not stop.is_set():
#         winsound.Beep(1500, 350)
#         stop.wait(.15)


# def main():
#     cam = cv2.VideoCapture(CAMERA)
#     if not cam.isOpened():
#         raise SystemExit("Webcam not available")

#     opt = vision.FaceLandmarkerOptions(
#         base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
#         running_mode=vision.RunningMode.VIDEO, num_faces=1)

#     closed = yawn = alarm_thread = stop = None
#     timestamp = 0

#     with vision.FaceLandmarker.create_from_options(opt) as detector:
#         while True:
#             ok, frame = cam.read()
#             if not ok: break

#             frame = cv2.flip(frame, 1)
#             h, w = frame.shape[:2]
#             image = mp.Image(
#                 mp.ImageFormat.SRGB,
#                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#             timestamp += 33
#             result = detector.detect_for_video(image, timestamp)

#             EAR = MAR = 0
#             status, alert = "NO FACE DETECTED", False

#             if result.face_landmarks:
#                 face = result.face_landmarks[0]
#                 EAR = float(np.mean([ear(points(face,e,w,h)) for e in EYES]))

#                 m = points(face, MOUTH, w, h)
#                 MAR = float(np.linalg.norm(m[2]-m[3]) /
#                             max(np.linalg.norm(m[0]-m[1]), 1))

#                 now = time.monotonic()
#                 if EAR < EAR_LIMIT and closed is None: closed = now
#                 if MAR > MAR_LIMIT and yawn is None: yawn = now
#                 if EAR >= EAR_LIMIT: closed = None
#                 if MAR <= MAR_LIMIT: yawn = None

#                 eye_alert = closed and now-closed >= EYE_TIME
#                 yawn_alert = yawn and now-yawn >= YAWN_TIME
#                 alert = bool(eye_alert or yawn_alert)
#                 status = ("DROWSINESS ALERT" if eye_alert else
#                           "YAWN DETECTED" if yawn_alert else "MONITORING")
#             else:
#                 closed = yawn = None

#             if alert and alarm_thread is None:
#                 stop = threading.Event()
#                 alarm_thread = threading.Thread(
#                     target=alarm, args=(stop,), daemon=True)
#                 alarm_thread.start()

#             if not alert and alarm_thread:
#                 stop.set()
#                 alarm_thread = stop = None

#             color = (60,70,255) if alert else (72,210,145)
#             cv2.putText(frame, f"EAR {EAR:.2f}  MAR {MAR:.2f}",
#                         (20,35), 0, .65, (255,255,255), 2)
#             cv2.putText(frame, status, (20,70),
#                         0, .65, color, 2)
#             cv2.imshow("Driver Drowsiness Monitor", frame)

#             if cv2.waitKey(1) & 255 in (ord("q"), 27): break

#         if stop: stop.set()
#         cam.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
