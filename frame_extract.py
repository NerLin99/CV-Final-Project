import cv2

if __name__ == '__main__':
    video_path = '/Users/22695/OneDrive/CV_Proj/spann3r/source_video/IMG_2534.MOV'
    save_path = '/Users/22695/OneDrive/CV_Proj/spann3r/examples/t2'
    video = cv2.VideoCapture(video_path)
    index = 0

    if video.isOpened():
        f = int(video.get(cv2.CAP_PROP_FPS))
        print('FPS: ', f)
        rval, frame = video.read()
    else:
        rval = False


    while rval:
        ret, frame = video.read()
        if frame is None:
            break
        else:
            cv2.imwrite(save_path + '/' + str('%04d' %index)+'.jpg', frame)
        index += 1