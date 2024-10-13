import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

from custom_gestures import recognize_tilt, recognize_fist_flexible

import pyautogui

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"  # Update this to the correct path where the model is saved, if not in current directory

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to a Mediapipe Image object for the gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            # TODO do the ad-hoc thing simultaneously?? 

            # Perform gesture recognition on the image
            result = gesture_recognizer.recognize(mp_image)

            resultALT = hands.process(image_rgb)

            # Draw the gesture recognition results on the image
            if result.gestures:
                recognized_gesture = result.gestures[0][0].category_name
                confidence = result.gestures[0][0].score

                tilt = None

                if resultALT.multi_hand_landmarks:
                    for hand_landmarks in resultALT.multi_hand_landmarks:
                        tilt = recognize_tilt(hand_landmarks, threshold_angle=25)

                # The conditionals in our main loop are based around a "heirarchy" of overlapping gesture
                # characteristics, allowing us to use unsophisticated gesture recognizers without registering 
                # unintuitive results. (1) The "victory" sign looks like no other gestures. (2)The canned gesture 
                #for closed_fist is strict, so we created recognize_fist_flexible, which is not strict at all
                # However the generous acceptance condition for recognize_fist_flexible does not overlap
                # with any of the acceptance conditions for thumb up/down/left/right, so there is no risk
                # of "preempting." (3) Thumb left/right share characteristics with thumb up/down. The 
                # tilt function to detect thumb left/right is stricter, only registering angles within 
                # 25 degrees of the "ideal" position. So it does not risk false positives. (4) Thumb up/down
                # can only plausible be confused with left/right. So if left/right are not recognized, than
                # a thumb up/down will not be a false positive. (5) If no gesture is recognized, no action
                # (as with closed fist)
                if recognized_gesture == "Victory":
                    print("space")
                    pyautogui.press("space")
                
                elif (recognized_gesture == "Closed_Fist") or recognize_fist_flexible(hand_landmarks): # rest position
                    print("no action")
                elif tilt == "LEFT":
                    print("Left")
                    pyautogui.press("left")
                elif tilt == "RIGHT":
                    print("right")
                    pyautogui.press("right")
                elif recognized_gesture == "Thumb_Up":
                    print("up")
                    pyautogui.press("up")
                elif recognized_gesture == "Thumb_Down":
                    print("down")
                    pyautogui.press("down")

                else: # rest position
                    print("no action")
                    pass


                # Display recognized gesture and confidence 
                cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the resulting image (can comment this out for better performance later on)
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()