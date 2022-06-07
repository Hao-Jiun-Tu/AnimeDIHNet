import numpy as np
import cv2
import random
import os
import sys
import time

if __name__ == '__main__':
    # video_to_image()
    filename = "../video/video0.mp4"
    out_path = "../video_data/video0"


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    cap = cv2.VideoCapture(filename)
    fps = int(cap.get(cv2.CAP_PROP_FPS) / 3)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)	# total frame
    current_frame = 0

    for i in range(int(total_frame)):
        ret = cap.grab()
        if ret:
            if i % fps == 0:
                ret, frame = cap.retrieve()
                if ret:
                    # if video is still left continue creating images
                    name = out_path + '/frame' +str(current_frame) + '.png'
                    print ('Creating...' + name)

                    # writing the extracted images
                    cv2.imwrite(name, frame)

                    # increasing counter so that it will
                    # show how many frames are created
                    current_frame += 1
                else:
                    print("Error retrieving frame from movie!")
                    break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()