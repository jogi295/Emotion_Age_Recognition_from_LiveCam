import cv2
import numpy as np
import tensorflow as tf


def avatar_on_image(base,emoji):
    base_image = np.array(base)
    emoji = np.array(emoji)
    if emoji.shape[2]==4:
        emoji = emoji[:,:,:3]
    
    (rows,cols,cha) = base_image.shape
    (r,c,ch) = emoji.shape
    
    emoji = cv2.resize(emoji,(180,200))
    base_image[:200, :180] = emoji[:,:]
    base_image = base_image.astype(np.uint8)
    
    return base_image



# load the video file; if video from cam replace file name with 0
cap = cv2.VideoCapture(0)

emotion_dic = {0:'Angry', 1:'Happy', 2:'Neutral', 3:'Sad'}
path0 = r'emojis/female/'
path1 = r'emojis/male/'
female_emoji_dic = {0:path0+'angry.png', 1:path0+'happy.png', 2:path0+'neutral.png', 3:path0+'sad.png'}
male_emoji_dic = {0:path1+'angry.png', 1:path1+'happy.png', 2:path1+'neutral.png', 3:path1+'sad.png'}
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Load and Initialize the  classifiers required
face_detector = cv2.CascadeClassifier('C:/Users/Vijay/Desktop/Jo/Applied_AI_Assignments/Emotion_detection_project/models/haarcascade_frontalface_default.xml')
emo_model = tf.keras.models.load_model('models/emo_model_modified_70.h5')
gen_model = tf.keras.models.load_model('models/gen_model_94.h5')


# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         # converts video frame into gray frame
        faces = face_detector.detectMultiScale(gray, 1.3, 5)   # detects four co-ordinates of the fa
     
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2) # create box using co-ordinates
            roi_frame = gray[y:y+h, x:x+w]
            resized_roi = cv2.resize(roi_frame, (48,48), interpolation = cv2.INTER_LINEAR)
            res = tf.expand_dims(resized_roi, 2)
            res1 = tf.expand_dims(res, 0)

            emo_predict = emo_model.predict(res1)              # it predicts the emotion from frame
            gen_predict = gen_model.predict(res1)              # it predicts the gender from frame

            maxindex_emo = int(np.argmax(emo_predict))         # get the maxindex from model prediction
            maxindex_gen = int(np.argmax(gen_predict))         # get the maxindex from model prediction
            
            # put text above the box
            cv2.putText(frame, emotion_dic[maxindex_emo],(x+20, y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            
            # emoji creation on the frame
            if maxindex_gen==0:
                emoji0 = cv2.imread(female_emoji_dic[maxindex_emo])
                frame = avatar_on_image(frame,emoji0) 
                cv2.imshow('Frame',frame)
            elif maxindex_gen==1:
                emoji1 = cv2.imread(male_emoji_dic[maxindex_emo])
                frame = avatar_on_image(frame,emoji1)
                cv2.imshow('Frame',frame)
    
    
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()             # when everything is done, release the video
cv2.destroyAllWindows()   # close all the windows opeded