from ursina import *
from ursina.prefabs.sky import Sky
from first_person_controller import FirstPersonController
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
from capture_webcam import calibrate_initial_pos, detect_blink, estimate_head_pose, determine_head_direction, draw_face_mesh

Text.default_font = 'fonts/LibertinusMono-Regular.ttf'
Text.default_resolution = 1920 * Text.size  # Set this BEFORE Ursina() and any UI entities

app = Ursina(size=(953,580))
window.borderless = False
window.position = Vec2(460,100)
Sky()

# Define a Voxel class.
# By setting the parent to scene and the model to 'cube' it becomes a 3d button.
last_highlighted_voxel = None
last_highlighted_artwork = None

class Voxel(Button):
    def __init__(self, position=(0,0,0)):
        super().__init__(parent=scene,
            position=position,
            model='cube',
            origin_y=.5,
            texture='white_cube',
            color=color.hsv(0, 0, random.uniform(.9, 1.0)),
        )

class WallVoxel(Button):
    def __init__(self, position=(0,0,0)):
        super().__init__(parent=scene,
            position=position,
            model='cube',
            origin_y=.5,
            texture='white_cube',
            color='#8aabba',  # blue-gray tint
            disabled=True
        )

class Artwork(Button):
    def __init__(self, title, model, artist, description, scale, position=(0,0,0)):
        super().__init__(parent=scene,
            model=model,
            position=position,
            scale=scale,
            color=color.white
        )
        self.title = title
        self.artist = artist
        self.description = description

# ~~~~~~~~~~~ SCENE OBJECTS ~~~~~~~~~~~
# Floor voxels
for z in range(16):
    for x in range(16):
        voxel = Voxel(position=(x,0,z))

# Wall height
wall_height = 6
# Walls at edges
for y in range(wall_height):
    # z=0 and z=15 (front/back)
    for x in range(16):
        WallVoxel(position=(x, y, 0))
        WallVoxel(position=(x, y, 15))
    # x=0 and x=15 (left/right)
    for z in range(16):
        WallVoxel(position=(0, y, z))
        WallVoxel(position=(15, y, z))
        
# Example for a GLTF model
starry_night = Artwork(
    title='"The Starry Night"',
    model='de_sterrennacht__nit_estelada__the_starry_night/scene.gltf',
    artist='Vincent van Gogh',
    description='The Starry Night is an oil on canvas by the Dutch post-impressionist painter Vincent van Gogh.',
    scale=3,
    position=(5,3,14)
)

torii_gate = Artwork(
    title='Torii Gate',
    model='japanese_torii_gate_game_asset.glb',
    artist='N/A',
    description='A traditional Japanese torii gate.',
    scale=0.008,
    position=(11,0,12)
)


player = FirstPersonController(position=(7.5,1.5,7.5))

# ~~~~~~~~~~~ GUI ~~~~~~~~~~~

# GUI panel for artwork info
artwork_panel = None
artwork_panel_bg = None
artwork_panel_open = False

# For blink-to-close logic
eyes_closed_start_time = None
eyes_were_closed = False

# Voice/LLM state
voice_thread = None
voice_active = False
voice_prompt = ''
llm_response = ''

VOICE_KEYWORD = 'art guide'  

def open_artwork_gui(artwork):
    global artwork_panel, artwork_panel_bg, artwork_panel_open, voice_thread, voice_active, voice_prompt, llm_response
    if artwork_panel_open:
        return
    player.on_disable()  # Lock cursor and disable movement
    artwork_panel_bg = Entity(parent=camera.ui, model='quad', color=color.black66, scale=(4, 4), z=-1)
    artwork_panel = Text(
        f"{artwork.title}\n{artwork.artist}\n\n{artwork.description}\n\nSay '{VOICE_KEYWORD}' to ask a question...",
        parent=artwork_panel_bg,
        origin=(0,0),   
        size=0.01,       
        wordwrap=40,
        color=color.white
    )
    artwork_panel_open = True
    # Start voice listener thread
    voice_active = True
    voice_prompt = ''
    llm_response = ''
    voice_thread = threading.Thread(target=voice_listener, args=(artwork,), daemon=True)
    voice_thread.start()

def close_artwork_gui():
    global artwork_panel, artwork_panel_bg, artwork_panel_open, voice_active
    if artwork_panel:
        destroy(artwork_panel)
        artwork_panel = None
    if artwork_panel_bg:
        destroy(artwork_panel_bg)
        artwork_panel_bg = None
    if artwork_panel_open:
        player.on_enable()  # Unlock cursor and enable movement
        artwork_panel_open = False
    voice_active = False  # Stop voice thread if open

# def input(key):
#     global artwork_panel_open
#     if artwork_panel_open and key in ('escape', 'right mouse down', 'left mouse down'):
#         close_artwork_gui()
#         return
#     if key == 'left mouse down':
#         hit_info = raycast(camera.world_position, camera.forward, distance=5)
#         if hit_info.hit:
#             Voxel(position=hit_info.entity.position + hit_info.normal)
#     if key == 'right mouse down' and mouse.hovered_entity:
#         destroy(mouse.hovered_entity)

# --- VOICE LISTENER AND LLM ---
def voice_listener(artwork):
    global voice_active, artwork_panel, voice_prompt, llm_response
    r = sr.Recognizer()
    mic = sr.Microphone()
    while voice_active:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio).lower()
            except sr.UnknownValueError:
                continue
            if VOICE_KEYWORD in text:
                if artwork_panel:
                    artwork_panel.text = f"{artwork.title}\n{artwork.artist}\n\nListening..."
                try:
                    with mic as source:
                        print("Listening for prompt...")
                        audio2 = r.listen(source, timeout=10)
                        print("Got audio, recognizing...")
                    try:
                        prompt = r.recognize_google(audio2)
                        print("You said:", prompt)
                    except sr.UnknownValueError:
                        prompt = "(Could not understand)"
                    except sr.RequestError as e:
                        prompt = f"(API error: {e})"
                    except Exception as e:
                        prompt = f"(Recognition error: {e})"
                except sr.WaitTimeoutError:
                    prompt = "(No speech detected)"
                except Exception as e:
                    prompt = f"(Mic error: {e})"
                # Update GUI with prompt
                if artwork_panel:
                    destroy(artwork_panel)
                    artwork_panel = Text(
                        f"{artwork.title}\n{artwork.artist}\n\nYou asked: {prompt}\n\nWaiting for answer...",
                        parent=artwork_panel_bg,
                            origin=(0,0),   
                            size=0.01,         
                            wordwrap=40,
                            color=color.white
                        )
                    print("Prompt set to:", prompt) 
                # Query LLM
                context = f"Artwork: {artwork.title} by {artwork.artist}. Description: {artwork.description}. User asked: {prompt}"
                try:
                    client = OpenAI(
                        api_key = os.getenv("OPENAI_API_KEY"),
                    )
                    response = client.responses.create(
                        model="gpt-4.1-mini",
                        instructions="You are an expert on art. \
                                Be concise with your responses (4-5 sentences maximum, unless you \
                                require more explanation). Be personable, and respond like a human guide would.",
                        input=context
                    )
                    llm_response = response.output_text
                    print("LLM response:", llm_response)
                except Exception as e:
                    llm_response = f"(Error: {e})"
                # Update GUI with response
                if artwork_panel:
                    destroy(artwork_panel)
                    artwork_panel = Text(
                        f"{artwork.title}\n{artwork.artist}\n\nYou asked: {prompt}\n\nAnswer: {llm_response}",
                        parent=artwork_panel_bg,
                        origin=(0,0),   
                        size=0.01,         
                        wordwrap=40,
                        color=color.white
                    )
                print("Response set to:", llm_response)
        except Exception:
            continue

# ~~~~~~~~~~~ WEBCAM ~~~~~~~~~~~
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Feed', 320, 240)
cv2.moveWindow('Webcam Feed', 0, 0)

neutral_dx, neutral_dy = None, None
prev_dx, prev_dy = 0, 0

# Webcam state variables
cap = None
face_mesh = None
neutral_dx = None
neutral_dy = None
calibrating = True
blink_total = 0
blink_counter = 0
blink_state = False
BLINK_EAR_THRESH = 0.10
BLINK_CONSEC_FRAMES = 2
LOOK_SENSITIVITY = 40

window_initialized = False

# Webcam setup/init
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    cap = None
print("Press 'q' to quit.")

def process_webcam_frame():
    global neutral_dx, neutral_dy, calibrating
    global blink_total, blink_counter, blink_state, blink_detected
    global ear, dx, dy
    if cap is None or face_mesh is None:
        return
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        return
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if calibrating:
        cv2.putText(frame, "Calibration: Look straight at the camera and press SPACE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)
        cv2.imshow('Webcam Feed', frame)
        key = cv2.waitKey(1)
        if key == 32 and results.multi_face_landmarks:
            # Calibrate neutral dx, dy
            face_landmarks = results.multi_face_landmarks[0]
            nose_tip = face_landmarks.landmark[1]
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
            neutral_dx = nose_tip.x - eye_center_x
            eye_center_y = (left_eye_outer.y + right_eye_outer.y) / 2
            neutral_dy = nose_tip.y - eye_center_y
            print(f"Calibration complete. Neutral dx: {neutral_dx:.4f}, dy: {neutral_dy:.4f}")
            calibrating = False
        elif key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            from ursina import application
            application.quit()
        return
    else:
        global window_initialized
        if not window_initialized:
            cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Webcam Feed', 662, 400)
            cv2.moveWindow('Webcam Feed', 0, 0)
            window_initialized = True
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            ear = detect_blink(face_landmarks, frame)
            dx, dy = estimate_head_pose(face_landmarks, neutral_dx, neutral_dy)
            draw_face_mesh(face_landmarks, frame)
            # --- Blink detection and teleportation ---
            blink_detected = False
            if ear < BLINK_EAR_THRESH:
                blink_counter += 1
                if blink_counter == BLINK_CONSEC_FRAMES and not blink_state:
                    blink_total += 1
                    blink_state = True
                    blink_detected = True
            else:
                blink_counter = 0
                blink_state = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame, f"Blinks: {blink_total}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255) if blink_state else (0,255,0), 3)

        # Map head direction to player look
        player.apply_webcam_look([dx * LOOK_SENSITIVITY, dy * LOOK_SENSITIVITY])

        # --- Teleport or interact on blink ---
        if blink_detected:
            hit_info = raycast(camera.world_position, camera.forward, distance=12)
            if hit_info.hit:
                if isinstance(hit_info.entity, Artwork):
                    open_artwork_gui(hit_info.entity)
                elif isinstance(hit_info.entity, Voxel) and not isinstance(hit_info.entity, WallVoxel):
                    # Teleport player to voxel position (stand on top)
                    target_pos = hit_info.entity.position + Vec3(0, 0, 0)
                    player.position = target_pos
        # --- Blink-to-close GUI logic ---
        global eyes_closed_start_time, eyes_were_closed
        if artwork_panel_open:
            if ear < BLINK_EAR_THRESH:
                if not eyes_were_closed:
                    eyes_closed_start_time = time.time()
                    eyes_were_closed = True
                elif eyes_closed_start_time and (time.time() - eyes_closed_start_time) >= 1.0:
                    close_artwork_gui()
                    eyes_closed_start_time = None
                    eyes_were_closed = False
            else:
                eyes_closed_start_time = None
                eyes_were_closed = False
        else:
            eyes_closed_start_time = None
            eyes_were_closed = False

        cv2.imshow('Webcam Feed', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            from ursina import application
            application.quit()


def update():
    process_webcam_frame()

    # At the end of process_webcam_frame or in update()
    global last_highlighted_voxel, last_highlighted_artwork

    hit_info = raycast(camera.world_position, camera.forward, distance=12)
    entity = hit_info.entity if hit_info.hit else None

    # Handle voxel highlighting
    if isinstance(entity, Voxel):
        if last_highlighted_voxel and last_highlighted_voxel != entity:
            last_highlighted_voxel.color = color.hsv(0, 0, random.uniform(.9, 1.0))
        entity.color = color.lime
        last_highlighted_voxel = entity
    else:
        if last_highlighted_voxel:
            last_highlighted_voxel.color = color.hsv(0, 0, random.uniform(.9, 1.0))
            last_highlighted_voxel = None

    # Handle artwork highlighting
    if isinstance(entity, Artwork):
        if last_highlighted_artwork and last_highlighted_artwork != entity:
            last_highlighted_artwork.color = color.white
        entity.highlight_color = color.white33
        last_highlighted_artwork = entity
    else:
        if last_highlighted_artwork:
            last_highlighted_artwork.color = color.white
            last_highlighted_artwork = None

app.run()