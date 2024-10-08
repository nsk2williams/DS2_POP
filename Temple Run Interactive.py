import cv2
import mediapipe as mp
import pyautogui

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"  # Update this to the correct path where the model is saved, if not in current directory

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

# Initialize MediaPipe hands module for detecting hand landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def detect_thumb_direction(landmarks):
    # Thumb tip and base coordinates
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]

    # Compare x-coordinates: thumb tip to thumb base
    if thumb_tip.x < thumb_cmc.x:  # Thumb pointing left
        return "Thumb_Left"
    elif thumb_tip.x > thumb_cmc.x:  # Thumb pointing right
        return "Thumb_Right"
    return None

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally and convert the BGR image to RGB.
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform hand landmark detection
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Detect thumb direction
                thumb_direction = detect_thumb_direction(hand_landmarks.landmark)

                if thumb_direction == "Thumb_Left":
                    pyautogui.press("left")
                elif thumb_direction == "Thumb_Right":
                    pyautogui.press("right")

                # Draw hand landmarks on the image (optional)
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert the image to a Mediapipe Image object for the gesture recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Perform gesture recognition on the image
        result = gesture_recognizer.recognize(mp_image)

        # Draw the gesture recognition results on the image
        if result.gestures:
            recognized_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score

            # Example of pressing keys with pyautogui based on recognized gesture
            if recognized_gesture == "Thumb_Up":
                pyautogui.press("up")
            elif recognized_gesture == "Thumb_Down":
                pyautogui.press("down")
            elif recognized_gesture == "Open_Palm":
                pyautogui.press("left")
            elif recognized_gesture == "Closed_Fist":
                pyautogui.press("right")
            elif recognized_gesture == "Victory":
                pyautogui.press("space")

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
