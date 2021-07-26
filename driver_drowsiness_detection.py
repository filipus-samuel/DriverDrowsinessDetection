from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream  # buat akses kamera

from imutils import face_utils
from imutils.video import FPS
import argparse  # command line arguments
import imutils  # resizing our images
import time  # for safety when opening a camera, giving time especially lower spec camera for processing the adjustment
import dlib  # for our face detector and facial landmark predictor
import cv2  # for opencv by windows
from time import sleep

# for alarm
import winsound
duration = 1200  # milliseconds
freq = 440  # Hz


def eye_aspect_ratio(eye):
    vertical_dist = dist.euclidean(
        eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])
    #print("vertical: ", vertical_dist)
    horizontal_dist = dist.euclidean(eye[0], eye[3])
    #print("horizontal: ", horizontal_dist)
    ear = (vertical_dist / (2.0 * horizontal_dist))
    #print("ear: ", ear)
    return ear


BLINK_THRESHOLD = 0

# get arguments from a command line
ap = argparse.ArgumentParser(description='Eye blink detection')
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
# instantiate HOG based and SVM Liner Face Detector
detector = dlib.get_frontal_face_detector()
# load our facial landmark predictor
predictor = dlib.shape_predictor(args["shape_predictor"])

# choose indexes for the left and right eye
(left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream or video reading fm the file
video_path = args["video"]
if video_path == "":
    vs = VideoStream(src=0).start()
    print("[INFO] starting video stream from built-in webcam...")
    fileStream = False
else:
    vs = FileVideoStream(video_path).start()
    print("[INFO] starting video stream from a file...")
    fileStream = True
time.sleep(1.0)

total = 0
alert = False
start_time = 0
frame = vs.read()
fps = FPS().start()
#ser = serial.Serial('COM6', 9600, timeout=1)

# adaptive threshold setting
# count start from 1
count_open = 1
count_closed = 1

# total from 0
ear_open_total = 0
ear_closed_total = 0

# 0 = tutup, 1 buka
clear_open = 0
clear_closed = 0
clear_open_closed = 0

# indicator kelar ngitung
done_counting = 0
pop_up_reset = 0

# sound
sound_success = 0
sound_reset = 0

while (not fileStream) or (frame is not None):
    # pre processing the frame that is being read by resizing it to the width of 640
    frame = imutils.resize(frame, width=640)

    # convert the frame into grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # passing the grayscale image into the HOG and SVM Linear face detector resulting in our bounding boxes
    rects = detector(gray_frame, 0)
    ear = 0

    fps.update()
    fps.stop()
    cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for rect in rects:
        shape = predictor(gray_frame, rect)
        # w which then we convert them from dlib object into a numpy array
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[left_s:left_e]  # only viewing our left eye
        rightEye = shape[right_s:right_e]  # only viewing our right eye
        leftEAR = eye_aspect_ratio(leftEye)  # computation of our EAR left eye
        # computation of our EAR right eye
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # adaptive threshold
        if clear_open == 1:
            if(count_open <= 10):
                count_open += 1
                ear_open_total += ear
            elif(count_open == 11):
                count_open += 1
                ear_open_total /= 10
                clear_open = 0

        if clear_closed == 1:
            if(count_closed <= 10):
                count_closed += 1
                ear_closed_total += ear
            elif(count_closed == 11):
                count_closed += 1
                ear_closed_total /= 10
                clear_closed = 0

        if clear_open_closed == 1:
            BLINK_THRESHOLD = (ear_open_total+ear_closed_total)/2
            print("NEW THRESHOLD: ", BLINK_THRESHOLD)
            clear_open_closed = 0
            done_counting = 1

        # if the eyes are closed longer than for 2 secs, raise an alert
        # if done counting the threshold, time to detect drowsy
        if ear < BLINK_THRESHOLD and done_counting == 1:
            if start_time == 0:
                start_time = time.time()
            else:
                end_time = time.time()
                if end_time - start_time > 2:
                    alert = True

        else:
            start_time = 0
            alert = False
            no_appereance = False

    cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # middle top side
    cv2.putText(frame, "Threshold: {:.2f}".format(BLINK_THRESHOLD), (250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # bottom left side
    cv2.putText(frame, "Steps to get your threshold:", (10, 435),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "1. Open your eyes and Press O to count Average EAR Open", (10, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.putText(frame, "2. Closed your eyes and Press C to count Average EAR Closed", (10, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.putText(frame, "3. After you've done step 1 and 2, Press T to count Threshold", (10, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.putText(frame, "4. If you want to make a new Threshold, Press R", (10, 475),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    # bottom right side
    cv2.putText(frame, "Average EAR Open : {:.2f}".format(ear_open_total), (450, 435),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "Average EAR Closed: {:.2f}".format(ear_closed_total), (450, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if pop_up_reset >= 1 and pop_up_reset <= 20:
        pop_up_reset += 1
        cv2.putText(frame, "Reset Threshold", (450, 475),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # if alert = 1, alarm bunyi
    if alert:
        cv2.putText(frame, "ALERT!", (250, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        winsound.Beep(freq, duration)

    # show the frame
    cv2.imshow("Driver Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # opening counting EAR open
    if key == ord("o"):
        clear_open = 1
        sound_success = 1

    # opening counting EAR closed
    if key == ord("c"):
        clear_closed = 1
        sound_success = 1

    # opening counting EAR threshold
    if key == ord("t"):
        clear_open_closed = 1
        sound_success = 1

    # Reset to take another new threshold
    if key == ord("r"):
        count_open = 1
        count_closed = 1
        ear_open_total = 0
        ear_closed_total = 0
        clear_open = 0
        clear_closed = 0
        clear_open_closed = 0
        done_counting = 0
        pop_up_reset = 1
        sound_reset = 1
        BLINK_THRESHOLD = 0
    frame = vs.read()
vs.stop()
