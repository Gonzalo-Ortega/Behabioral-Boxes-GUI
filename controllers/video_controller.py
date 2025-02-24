import time
import numpy as np
import imutils
import cv2
import datetime
import os


def run_video(running, video_num, file_name, isRecording, Xmean, Ymean, XTA, YTA, RTA, Xport, Yport, Xcirc, Ycirc, Rcirc):
    h = None
    print(f'[VIDEO]: {video_num}')
    video_cap = cv2.VideoCapture(video_num, cv2.CAP_DSHOW)
    time.sleep(1.0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    valueTime = datetime.datetime.fromtimestamp(time.time())
    filename = valueTime.strftime(file_name)
    path = 'video_output'
    pathname = os.path.join(path, filename)
    print('[OUT] Create video file: ' + pathname)
    out = cv2.VideoWriter(pathname, fourcc, 20, (320, 240))  # cv2.VideoWriter(pathname,fourcc, 20, (640,480))
    fgbg = cv2.createBackgroundSubtractorKNN()
    Xm = np.ones(8)
    Xm[0:8] = Xport.value
    Ym = np.ones(8)
    Ym[0:8] = Yport.value

    while running.value == 1:
        ret2, frame = video_cap.read()

        # Write the output frame to file
        if h is None:
            (h, w) = frame.shape[:2]
        frameSmall = imutils.resize(frame, width=320)

        # Write the flipped frame
        if isRecording.value == 1:
            out.write(frameSmall)

        # Converting frames in a gray scale picture, thresholding and subtracting the foreground
        frameBW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouseColor = 0.45 * np.percentile(np.concatenate(frameBW), 10) + 15

        # Max value for filter is 21
        frameBlur = frameBW
        foreGround = fgbg.apply(frameBW)
        tresholdMask = cv2.threshold(frameBlur, int(mouseColor), 255, cv2.THRESH_BINARY)[1]  # threshold 80\
        foreGround[np.nonzero(tresholdMask)] = 0
        Ymask, Xmask = np.nonzero(foreGround)
        if 0 < len(Xmask) and 0 < len(Ymask):
            if (np.sqrt(np.square(int(np.floor(np.mean(Xmask))) - Xcirc.value) + np.square(
                    int(np.floor(np.mean(Ymask))) - Ycirc.value)) < Rcirc.value * 1.02) and (
                    np.sqrt(np.square((int(np.floor(np.mean(Xmask))) - int(np.floor(np.mean(Xm))))) + np.square(
                            (int(np.floor(np.mean(Ymask))) - int(np.floor(np.mean(Ym)))))) < 150):
                Xmean.value = int(np.floor(np.mean(Xmask)))  # int(np.floor(np.mean((cnts[1][0].T)[0,0])));
                Ymean.value = int(np.floor(np.mean(Ymask)))  # int(np.floor(np.mean((cnts[1][0].T)[1,0])));
                Xm[7] = Xm[6]
                Ym[7] = Ym[6]
                Xm[6] = Xm[5]
                Ym[6] = Ym[5]
                Xm[5] = Xm[4]
                Ym[5] = Ym[4]
                Xm[4] = Xm[3]
                Ym[4] = Ym[3]
                Xm[3] = Xm[2]
                Ym[3] = Ym[2]
                Xm[2] = Xm[1]
                Ym[2] = Ym[1]
                Xm[1] = Xm[0]
                Ym[1] = Ym[0]
                Xm[0] = int(np.floor(np.mean(Xmask)))
                Ym[0] = int(np.floor(np.mean(Ymask)))
        # plot a circle around the center point of the contour
        cv2.circle(frame, (Xmean.value, Ymean.value), 25, (255, 0, 255), 1, 8, 0)
        cv2.circle(frame, (XTA.value, YTA.value), int(round(RTA.value)), (0, 0, 255), 1, 14, 0)
        if isRecording.value == 1:
            cv2.circle(frame, (40, 440), 5, (0, 0, 255), 8, 8, 0)
        cv2.circle(frame, (Xport.value, Yport.value), 3, (0, 255, 0), 5, 8, 0)
        # show the frames
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if running.value == 0:
            break
    # Release everything if job is finished
    video_cap.release()
    out.release()
    cv2.destroyAllWindows()
