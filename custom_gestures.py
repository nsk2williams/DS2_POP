import cv2
import mediapipe as mp
import math
from math import pi

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])


# A simple function to broaden the conditions under which our system will recognize a fist (our resting gesture).
# The canned fist gesture often does not work at different angles, and especially does not work on the back of
# the hand, which is a natural resting position between our other gestures. This function checks whether the 
#thumb tip is touching, or very close to touching, the index knuckle. It is not a strict fist recognizer, and 
# probably has plenty of false positives. But none of the other gestures (pause, up/down/left/right) involve 
# putting the thumb anywhere near the index knuckle, so for our purposes, this simple heuristic is adequate. 
def recognize_fist_flexible(hand_landmarks):
    # Extract necessary landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    # Calculate distance between thumb tip and index knuckle
    distance = calculate_distance(
        (thumb_tip.x, thumb_tip.y), 
        (index_knuckle.x, index_knuckle.y)
    )

    if distance < 0.1:
        return True

    return False


# [Depcreated] First attempt to recognize tilt: based on whether thumb tip was to left or right of index knuckle
def recognize_tilt_basic(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    if index_knuckle.x - thumb_tip.x > 0.05:
        return "BEARING LEFT"
    elif index_knuckle.x - thumb_tip.x < -0.05:
        return "BEARING RIGHT"
    else:
        return "NEITHER"

# [Deprecated] Second attempt to recognize tilt: based on angle between thumb and index knuckle. Allows us to set "strictness"
# By giving an angle threshold past which gesture will not register (e.g. 35 degress of the axis)
def recognize_tilt_classic(hand_landmarks, threshold_angle=30):

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    pinky_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]


    angle = (math.atan((thumb_tip.y-index_knuckle.y) / (thumb_tip.x-index_knuckle.x))) * 180/pi
    #print("AP", round(angle_proxy))


    if (index_knuckle.x - thumb_tip.x > 0.05) and abs(angle) < (90-threshold_angle):
        return "LEFT" # + str(angle)
    elif (index_knuckle.x - thumb_tip.x < -0.05) and abs(angle) < (90-threshold_angle):
        return "RIGHT" # + str(angle)
    else:
        return "NEITHER" # + str(angle)


# The same as recognize_tilt_classic, but averages the position of index, middle, ring, and pinky knuckles. 
# When testing on different group members, we noticed that different hands have different natural shapes, and
# some people make a thumbs up with their thumb naturally leaning in one direction. This leads to some sharply
# "biased" angles between the index knuckle and thumb tip. Because the effect varied between different hands, 
# we could not counteract it directly (e.g. fixing it for right handed users might worsen it for left-handed
# users). So instead we "softened" the effect by taking a different angle, using the avreage positions of all
# non-thumb knuckles. This, coupled with a slightly tighter range of "forgiveness" (25 degrees) produced
# intuitive behavior which did not surprise users
def recognize_tilt(hand_landmarks, threshold_angle=30):

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    index_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    avg_knuckle_x = (index_knuckle.x + middle_knuckle.x + ring_knuckle.x + pinky_knuckle.x) /4
    avg_knuckle_y = (index_knuckle.y + middle_knuckle.y + ring_knuckle.y + pinky_knuckle.y) /4


    angle = (math.atan((thumb_tip.y-avg_knuckle_y) / (thumb_tip.x-avg_knuckle_x))) * 180/3.14


    if (index_knuckle.x - thumb_tip.x > 0.05) and abs(angle) < (90-threshold_angle):
        return "LEFT" # + str(angle)
    elif (index_knuckle.x - thumb_tip.x < -0.05) and abs(angle) < (90-threshold_angle):
        return "RIGHT" # + str(angle)
    else:
        return "NEITHER" # + str(angle)



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

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            # Draw the hand annotations on the image.
            # image_rgb.flags.writeable = True
            # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = recognize_tilt(hand_landmarks)
                    print("Gesture:", gesture)

                # for hand_landmarks in results.multi_hand_landmarks:
                #     # Draw landmarks
                #     mp_drawing.draw_landmarks(
                #         image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


                    
                #     # Display gesture near hand location
                #     cv2.putText(image, gesture, 
                #                 (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                #                  int(hand_landmarks.landmark[0].y * image.shape[0]) - 20),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the resulting image
            # cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
