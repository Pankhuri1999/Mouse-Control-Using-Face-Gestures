from imutils import face_utils
from utils import *
import numpy as np
import pyautogui as pag
import imutils
import dlib
import cv2

# Thresholds and consecutive frame length for triggering the mouse action.
mouthARThresh = 0.6
mouthFrames = 15
eyeARThresh = 0.19
eyeFrame = 15
winkThresh = 0.04
winkCloseThresh = 0.19
winkConsecutiveFrame = 10

# Initialize the frame counters for each action as well as 
# booleans used to indicate if action is performed or not
mouthCnt = 0
eyeCnt = 0
winkCnt = 0
inputRead = False
clickEye = False
winkLeft = False
winkRight = False
scrollStart = False
anchorPnt = (0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
black = (0, 0, 0)

# Initialize Dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shapePredict = "dataset/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredict)

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(noseStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mouthStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Video capture
vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h

while True:
    # Grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face in the grayscale frame
    face = detector(gray, 0)

    # Loop over the face detections
    if len(face) > 0:
        rect = face[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # Determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    mouth = shape[mouthStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[noseStart:nEnd]

    # Because I flipped the frame, left is right, right is left.
    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    # Average the mouth aspect ratio together for both eyes
    mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    nose_point = (nose[3, 0], nose[3, 1])

    # Compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, yellow, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, yellow, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, yellow, 1)

    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
        cv2.circle(frame, (x, y), 2, green, -1)
        
    # Check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if diff_ear > winkThresh:

        if leftEAR < rightEAR:
            if leftEAR < eyeARThresh:
                winkCnt += 1

                if winkCnt > winkConsecutiveFrame:
                    pag.click(button='left')

                    winkCnt = 0

        elif leftEAR > rightEAR:
            if rightEAR < eyeARThresh:
                winkCnt += 1

                if winkCnt > winkConsecutiveFrame:
                    pag.click(button='right')

                    winkCnt = 0
        else:
            winkCnt = 0
    else:
        if ear <= eyeARThresh:
            eyeCnt += 1

            if eyeCnt > eyeFrame:
                scrollStart = not scrollStart
                # inputRead = not inputRead
                eyeCnt = 0

                # nose point to draw a bounding box around it

        else:
            eyeCnt = 0
            winkCnt = 0

    if mar > mouthARThresh:
        mouthCnt += 1

        if mouthCnt >= mouthFrames:
            # if the alarm is not on, turn it on
            inputRead = not inputRead
            # scrollStart = not scrollStart
            mouthCnt = 0
            anchorPnt = nose_point

    else:
        mouthCnt = 0

    if inputRead:
        cv2.putText(frame, "Application has started reading input :) ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
        x, y = anchorPnt
        nx, ny = nose_point
        w, h = 60, 35
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), green, 2)
        cv2.line(frame, anchorPnt, nose_point, blue, 2)

        dir = direction(nose_point, anchorPnt, w, h)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
        drag = 18
        if dir == 'right':
            pag.moveRel(drag, 0)
        elif dir == 'left':
            pag.moveRel(-drag, 0)
        elif dir == 'up':
            if scrollStart:
                pag.scroll(40)
            else:
                pag.moveRel(0, -drag)
        elif dir == 'down':
            if scrollStart:
                pag.scroll(-40)
            else:
                pag.moveRel(0, drag)

    if scrollStart:
        cv2.putText(frame, 'Scroll mode is active ', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)

    # cv2.putText(frame, "MAR: {:.2f}".format(mar), (500, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2)
    # cv2.putText(frame, "Right EAR: {:.2f}".format(rightEAR), (460, 80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2)
    # cv2.putText(frame, "Left EAR: {:.2f}".format(leftEAR), (460, 130),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2)
    # cv2.putText(frame, "Diff EAR: {:.2f}".format(np.abs(leftEAR - rightEAR)), (460, 80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `Esc` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Do a bit of cleanup
cv2.destroyAllWindows()
vid.release()
