import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Drawing specifications for landmarks
DRAW_SPEC = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

def calibrate_initial_pos(cap, face_mesh):
    # --- Calibration step for head direction ---
    print("Calibration: Please look straight at the camera, and then press SPACE.")
    neutral_dx, neutral_dy = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            nose_tip = face_landmarks.landmark[1]
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
            dx = nose_tip.x - eye_center_x
            eye_center_y = (left_eye_outer.y + right_eye_outer.y) / 2
            dy = nose_tip.y - eye_center_y
            # Draw calibration info
            cv2.putText(frame, "Calibration: Look straight at the camera and press SPACE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 3)
        cv2.imshow('Webcam Feed', frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            if results.multi_face_landmarks:
                neutral_dx = dx
                neutral_dy = dy
                print(f"Calibration complete. Neutral dx: {neutral_dx:.4f}, dy: {neutral_dy:.4f}")
                break
        elif key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    # --- End calibration ---
    return neutral_dx, neutral_dy

def detect_blink(face_landmarks, frame):
    # EAR blink detection
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    def euclidean_dist(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5
    def eye_aspect_ratio(eye_landmarks):
        A = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
        B = euclidean_dist(eye_landmarks[2], eye_landmarks[4])
        C = euclidean_dist(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)
    ih, iw, _ = frame.shape
    left_eye = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in LEFT_EYE_IDX]
    right_eye = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in RIGHT_EYE_IDX]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    # Draw yellow dots on EAR landmarks after mesh for visibility
    for (x, y) in left_eye + right_eye:
        cv2.circle(frame, (x, y), 5, (0, 255, 255), 2)
    return ear

def estimate_head_pose(face_landmarks, neutral_dx, neutral_dy):
    # --- Head pose estimation (direction) ---
    # Use nose tip and eye corners for simple heuristic
    nose_tip = face_landmarks.landmark[1]
    left_eye_outer = face_landmarks.landmark[33]
    right_eye_outer = face_landmarks.landmark[263]
    # Horizontal: nose x vs. eye center x
    eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
    dx = nose_tip.x - eye_center_x - neutral_dx
    # Vertical: nose y vs. eye center y
    eye_center_y = (left_eye_outer.y + right_eye_outer.y) / 2
    dy = nose_tip.y - eye_center_y - neutral_dy
    return dx, dy

def determine_head_direction(dx, dy):
    # --- Head direction classification ---
    THRESH_X = 0.03
    THRESH_Y = 0.03
    if dx > THRESH_X:
        direction = "RIGHT"
    elif dx < -THRESH_X:
        direction = "LEFT"
    elif dy > THRESH_Y:
        direction = "DOWN"
    elif dy < -THRESH_Y:
        direction = "UP"
    else:
        direction = "CENTER"
    return direction

def draw_face_mesh(face_landmarks, frame):
    # Draw face mesh
    mp_drawing.draw_landmarks(
        frame,
        face_landmarks,
        mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=DRAW_SPEC
    )
    mp_drawing.draw_landmarks(
        frame,
        face_landmarks,
        mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=DRAW_SPEC,
        connection_drawing_spec=DRAW_SPEC
    )
    mp_drawing.draw_landmarks(
        frame,
        face_landmarks,
        mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=DRAW_SPEC,
        connection_drawing_spec=DRAW_SPEC
    )

def track_blinks(face_landmarks, ear):
    # --- Blink tracking ---
    BLINK_EAR_THRESH = 0.10
    BLINK_CONSEC_FRAMES = 2
    if not hasattr(main, 'blink_counter'):
        main.blink_counter = 0
        main.blink_total = 0
        main.blink_state = False
    if ear < BLINK_EAR_THRESH:
        main.blink_counter += 1
        if main.blink_counter == BLINK_CONSEC_FRAMES:
            main.blink_total += 1
            main.blink_state = True
    else:
        main.blink_counter = 0
        main.blink_state = False

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Press 'q' to quit.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        neutral_dx, neutral_dy = calibrate_initial_pos(cap, face_mesh)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)  # Mirror the camera horizontally
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ear = detect_blink(face_landmarks, frame)
                dx, dy = estimate_head_pose(face_landmarks, neutral_dx, neutral_dy)
                direction = determine_head_direction(dx, dy)
                draw_face_mesh(face_landmarks, frame)
                track_blinks(face_landmarks, ear)

                # Display tracking statuses
                cv2.putText(frame, f"Blinks: {main.blink_total}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255) if main.blink_state else (0,255,0), 3)
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                # Display head direction
                cv2.putText(frame, f"Facing: {direction}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,128,0), 3)
            cv2.imshow('Webcam Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):   
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
