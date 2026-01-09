import cv2
from filters import apply_filter
from gesture_control import detect_gesture

cap = cv2.VideoCapture(0)

current_filter = "original"   # default filter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gesture = detect_gesture(frame)
    print("Gesture:", gesture)

    if gesture == "1_fingers":
        current_filter = "original"

    elif gesture == "2_fingers":
        current_filter = "grey"

    elif gesture == "3_fingers":
        current_filter = "blur"

    elif gesture == "4_fingers":
        current_filter = "sepia"

    elif gesture == "5_fingers":
        current_filter = "edges"

    filtered = apply_filter(frame, current_filter)

    if len(filtered.shape) == 2:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        filtered,
        f"Current Filter: {current_filter}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Controlled Filters", filtered)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
