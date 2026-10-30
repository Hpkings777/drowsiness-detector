import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Save original pygame.draw reference to prevent infinite recursion
_ORIG_DRAW = pygame.draw

# ==========================================================
# CONFIGURATION
# ==========================================================

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 650 
RENDER_SCALE = 2.0             # High-definition internal resolution multiplier (1800x1300)
CAMERA_INDEX = 0
TARGET_FPS = 30

# Speed Limit & x/2 Physics Configuration (KM/H)
SPEED_LIMIT_KPH = 70.0         # Maximum speed shown on speedometer (x)
ACTUAL_SPEED_RATIO = 0.5       # If speedometer shows x, actual speed is x/2
SPEED_SCALE = 600.0            # Conversion factor between KM/H and world movement

# Eye-closure thresholds
EYE_CLOSED_THRESHOLD = 0.55
EYE_OPEN_THRESHOLD = 0.30

DROWSINESS_WARNING_LIMIT = 1.0  # Displays red screen warning
ALARM_CLOSURE_LIMIT = 2.0       # Triggers alarm audio & emergency braking

# How quickly the beep interval decreases over time
INITIAL_BEEP_INTERVAL = 1.00
MIN_BEEP_INTERVAL = 0.12
BEEP_INTERVAL_REDUCTION = 0.35

BEEP_DURATION = 0.12
BEEP_FREQUENCY = 1000

# Driver visibility
DRIVER_MISSING_LIMIT = 1.0

# MediaPipe model
MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME


# ==========================================================
# HIGH-RESOLUTION SCALED CANVAS WRAPPER
# ==========================================================

class HiResCanvas:
    """Renders all 900x650 drawing calls natively at high resolution."""
    def __init__(self, width, height, scale):
        self.s = scale
        self.surface = pygame.Surface((int(width * scale), int(height * scale)))

    def _pt(self, pt):
        return (int(round(pt[0] * self.s)), int(round(pt[1] * self.s)))

    def _rect(self, r):
        if isinstance(r, pygame.Rect):
            return pygame.Rect(int(round(r.x * self.s)), int(round(r.y * self.s)), int(round(r.width * self.s)), int(round(r.height * self.s)))
        return pygame.Rect(int(round(r[0] * self.s)), int(round(r[1] * self.s)), int(round(r[2] * self.s)), int(round(r[3] * self.s)))

    def _w(self, w):
        return 0 if w == 0 else max(1, int(round(w * self.s)))

    def blit(self, src, dest):
        if isinstance(dest, pygame.Rect):
            pos = (int(round(dest.x * self.s)), int(round(dest.y * self.s)))
        else:
            pos = self._pt(dest)
        scaled_src = pygame.transform.smoothscale(
            src, (max(1, int(round(src.get_width() * self.s))), max(1, int(round(src.get_height() * self.s))))
        )
        self.surface.blit(scaled_src, pos)


class HiResDraw:
    @staticmethod
    def polygon(target, color, points, width=0):
        if isinstance(target, HiResCanvas):
            pts = [target._pt(p) for p in points]
            return _ORIG_DRAW.polygon(target.surface, color, pts, target._w(width))
        return _ORIG_DRAW.polygon(target, color, points, width)

    @staticmethod
    def rect(target, color, rect, width=0, border_radius=-1):
        if isinstance(target, HiResCanvas):
            br = int(round(border_radius * target.s)) if border_radius > 0 else border_radius
            return _ORIG_DRAW.rect(target.surface, color, target._rect(rect), target._w(width), border_radius=br)
        return _ORIG_DRAW.rect(target, color, rect, width, border_radius=border_radius)

    @staticmethod
    def line(target, color, start_pos, end_pos, width=1):
        if isinstance(target, HiResCanvas):
            return _ORIG_DRAW.line(target.surface, color, target._pt(start_pos), target._pt(end_pos), target._w(width))
        return _ORIG_DRAW.line(target, color, start_pos, end_pos, width)

    @staticmethod
    def lines(target, color, closed, points, width=1):
        if isinstance(target, HiResCanvas):
            pts = [target._pt(p) for p in points]
            return _ORIG_DRAW.lines(target.surface, color, closed, pts, target._w(width))
        return _ORIG_DRAW.lines(target, color, closed, points, width)

    @staticmethod
    def circle(target, color, center, radius, width=0):
        if isinstance(target, HiResCanvas):
            return _ORIG_DRAW.circle(target.surface, color, target._pt(center), max(1, int(round(radius * target.s))), target._w(width))
        return _ORIG_DRAW.circle(target, color, center, radius, width)

    @staticmethod
    def ellipse(target, color, rect, width=0):
        if isinstance(target, HiResCanvas):
            return _ORIG_DRAW.ellipse(target.surface, color, target._rect(rect), target._w(width))
        return _ORIG_DRAW.ellipse(target, color, rect, width)


# ==========================================================
# DOWNLOAD MEDIAPIPE MODEL
# ==========================================================

def download_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print("MediaPipe model already exists.")
        return

    print("Downloading MediaPipe face model...")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError("Model download is incomplete.")
    print("Model downloaded successfully.")


# ==========================================================
# CREATE BEEP SOUND
# ==========================================================

def create_alarm_sound():
    sample_rate = 22050
    sample_count = int(sample_rate * BEEP_DURATION)
    t = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = (
        0.45 * np.sin(2 * np.pi * BEEP_FREQUENCY * t) * 32767
    ).astype(np.int16)

    fade_samples = max(1, int(sample_rate * 0.01))
    waveform[:fade_samples] = (
        waveform[:fade_samples] * np.linspace(0, 1, fade_samples, dtype=np.float32)
    ).astype(np.int16)
    waveform[-fade_samples:] = (
        waveform[-fade_samples:] * np.linspace(1, 0, fade_samples, dtype=np.float32)
    ).astype(np.int16)

    stereo_waveform = np.ascontiguousarray(
        np.column_stack((waveform, waveform))
    )
    return pygame.mixer.Sound(buffer=stereo_waveform.tobytes())


def get_beep_interval(duration_over_limit):
    interval = INITIAL_BEEP_INTERVAL - (BEEP_INTERVAL_REDUCTION * duration_over_limit)
    return max(MIN_BEEP_INTERVAL, interval)


def open_camera():
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if camera.isOpened():
        return camera
    camera.release()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if camera.isOpened():
        return camera
    camera.release()
    raise RuntimeError("Cannot open webcam. Check camera permissions.")


# ==========================================================
# PROCEDURAL GRAPHICS HELPERS
# ==========================================================

def create_sky_surface(width, horizon_y):
    sky = pygame.Surface((width, horizon_y))
    top_color = np.array([18, 32, 72], dtype=np.float32)
    mid_color = np.array([55, 115, 185], dtype=np.float32)
    bot_color = np.array([235, 185, 140], dtype=np.float32)

    for y in range(horizon_y):
        ratio = y / max(1, horizon_y - 1)
        if ratio < 0.65:
            t = ratio / 0.65
            c = top_color * (1 - t) + mid_color * t
        else:
            t = (ratio - 0.65) / 0.35
            c = mid_color * (1 - t) + bot_color * t
        pygame.draw.line(sky, c.astype(int).tolist(), (0, y), (width, y))

    sun_pos = (int(width * 0.72), int(horizon_y * 0.58))
    for radius, alpha in [(85, 25), (55, 55), (35, 110), (22, 255)]:
        glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 235, 180, alpha), (radius, radius), radius)
        sky.blit(glow, (sun_pos[0] - radius, sun_pos[1] - radius))

    return sky


def draw_scenery_object(screen, obj_type, x, y, scale):
    if scale < 0.04:
        return

    if obj_type == "tree":
        trunk_w = max(2, int(18 * scale))
        trunk_h = max(4, int(45 * scale))
        pygame.draw.rect(screen, (75, 48, 28), (x - trunk_w // 2, y - trunk_h, trunk_w, trunk_h))

        layers = [
            (int(75 * scale), int(85 * scale), int(30 * scale), (25, 85, 38), (40, 115, 52)),
            (int(60 * scale), int(75 * scale), int(65 * scale), (30, 100, 44), (50, 130, 60)),
            (int(42 * scale), int(60 * scale), int(100 * scale), (38, 118, 52), (60, 148, 70)),
        ]
        for half_w, h, y_off, dark_c, light_c in layers:
            base_y = y - y_off
            top_y = base_y - h
            pygame.draw.polygon(screen, dark_c, [(x, top_y), (x - half_w, base_y), (x, base_y)])
            pygame.draw.polygon(screen, light_c, [(x, top_y), (x + half_w, base_y), (x, base_y)])

    elif obj_type == "lamp":
        pole_h = max(8, int(210 * scale))
        pole_w = max(2, int(7 * scale))
        arm_w = max(4, int(55 * scale))
        direction = -1 if x > WINDOW_WIDTH // 2 else 1

        pygame.draw.line(screen, (90, 98, 110), (x, y), (x, y - pole_h), pole_w)
        arm_end_x = x + direction * arm_w
        arm_y = y - pole_h
        pygame.draw.line(screen, (110, 118, 130), (x, arm_y), (arm_end_x, arm_y), max(1, pole_w - 1))

        lamp_r = max(2, int(10 * scale))
        pygame.draw.circle(screen, (255, 245, 200), (arm_end_x, arm_y), lamp_r)


def draw_analog_gauge(screen, center, radius, speed_kph, max_kph, hud_font, small_font):
    cx, cy = center
    pygame.draw.circle(screen, (12, 16, 24), center, radius)
    pygame.draw.circle(screen, (45, 52, 65), center, radius, 6)
    pygame.draw.circle(screen, (110, 125, 145), center, radius - 4, 2)

    start_deg = 225.0
    total_span = 270.0

    for step in range(0, max_kph + 1, 10):
        ratio = step / max_kph
        deg = start_deg - ratio * total_span
        rad = math.radians(deg)
        is_major = (step % 20 == 0) or (step == int(SPEED_LIMIT_KPH))
        inner_r = radius - (16 if is_major else 10)
        outer_r = radius - 6

        x1 = cx + int(inner_r * math.cos(rad))
        y1 = cy - int(inner_r * math.sin(rad))
        x2 = cx + int(outer_r * math.cos(rad))
        y2 = cy - int(outer_r * math.sin(rad))

        tick_color = (255, 75, 75) if step >= int(SPEED_LIMIT_KPH) else (200, 215, 235)
        pygame.draw.line(screen, tick_color, (x1, y1), (x2, y2), 3 if is_major else 1)

    clamped_speed = max(0.0, min(float(max_kph), float(speed_kph)))
    needle_deg = start_deg - (clamped_speed / max_kph) * total_span
    n_rad = math.radians(needle_deg)
    nx = cx + int((radius - 14) * math.cos(n_rad))
    ny = cy - int((radius - 14) * math.sin(n_rad))
    pygame.draw.line(screen, (255, 65, 55), (cx, cy), (nx, ny), 4)
    pygame.draw.circle(screen, (35, 40, 50), center, 9)
    pygame.draw.circle(screen, (180, 190, 205), center, 9, 2)

    num_surf = hud_font.render(f"{int(round(speed_kph))}", True, (240, 250, 255))
    lbl_surf = small_font.render("KM/H", True, (130, 170, 210))
    screen.blit(num_surf, num_surf.get_rect(center=(cx, cy + 20)))
    screen.blit(lbl_surf, lbl_surf.get_rect(center=(cx, cy + 40)))


def draw_speed_limit_sign(screen, center, limit_value, font, sub_font):
    pygame.draw.circle(screen, (245, 245, 250), center, 26)
    pygame.draw.circle(screen, (225, 35, 35), center, 26, 6)
    num_txt = font.render(str(int(limit_value)), True, (20, 20, 25))
    lbl_txt = sub_font.render("LIMIT", True, (180, 195, 215))
    screen.blit(num_txt, num_txt.get_rect(center=center))
    screen.blit(lbl_txt, lbl_txt.get_rect(center=(center[0], center[1] + 36)))


def main():
    download_model()
    pygame.init()
    pygame.joystick.init()

    camera = None
    landmarker = None
    alarm_sound = None
    alarm_channel = None
    wheel = None

    if pygame.joystick.get_count() > 0:
        wheel = pygame.joystick.Joystick(0)
        wheel.init()
        print(f"Connected Wheel: {wheel.get_name()}")

    try:
        info = pygame.display.Info()
        monitor_width, monitor_height = info.current_w, info.current_h
        
        screen = pygame.display.set_mode(
            (monitor_width, monitor_height),
            pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        canvas = HiResCanvas(WINDOW_WIDTH, WINDOW_HEIGHT, RENDER_SCALE)
        pygame.draw = HiResDraw
        
        pygame.display.set_caption("Highway FPV Driving System (High Resolution)")
        clock = pygame.time.Clock()

        alert_font = pygame.font.SysFont("Arial", 20, bold=True)
        warning_overlay_font = pygame.font.SysFont("Arial", 40, bold=True)
        game_font = pygame.font.SysFont("Arial", 13, bold=True)
        hud_font = pygame.font.SysFont("Arial", 22, bold=True)

        horizon_y = 210
        sky_surface = create_sky_surface(WINDOW_WIDTH, horizon_y)
        vignette_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(vignette_surf, (0, 0, 0, 45), (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 18)

        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            alarm_sound = create_alarm_sound()
        except pygame.error as error:
            print("Audio initialization failed:", error)

        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        camera = open_camera()

        print("Highway FPV System started in High Resolution Fullscreen. Press Q or ESC to exit.")

        start_time = time.monotonic()
        last_timestamp_ms = -1

        eyes_are_closing = False
        closure_started_at = None
        current_closure_duration = 0.0

        face_missing_since = None
        alarm_active = False
        alarm_triggered_at = None
        next_beep_at = None

        player_x = 0.0
        target_player_x = 0.0
        
        displayed_speed_kph = 0.0
        max_displayed_kph = SPEED_LIMIT_KPH
        slow_displayed_kph = 30.0
        min_displayed_kph = 0.0

        is_braking = False
        braking_start_speed = 0.0
        braking_start_time = 0.0
        
        wheel_angle = 0.0
        world_z = 0.0
        game_score = 0.0

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key in (K_q, K_ESCAPE):
                        running = False
                    elif event.key == K_DOWN:
                        if not is_braking and displayed_speed_kph > 0.0:
                            is_braking = True
                            braking_start_speed = displayed_speed_kph
                            braking_start_time = time.monotonic()

            if not running:
                break

            success, frame = camera.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            now = time.monotonic()
            elapsed = now - start_time
            timestamp_ms = max(int(elapsed * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            face_detected = bool(result.face_landmarks)
            analysis_available = face_detected and bool(result.face_blendshapes)

            eye_crop = None
            both_eyes_closed = False
            both_eyes_open = False

            if not face_detected:
                if face_missing_since is None:
                    face_missing_since = now
            else:
                face_missing_since = None

            driver_not_visible = (
                face_missing_since is not None
                and now - face_missing_since >= DRIVER_MISSING_LIMIT
            )

            if analysis_available:
                blendshapes = result.face_blendshapes[0]
                scores = {item.category_name: item.score for item in blendshapes}
                left_closure = scores.get("eyeBlinkLeft", 0.0)
                right_closure = scores.get("eyeBlinkRight", 0.0)

                both_eyes_closed = (
                    left_closure >= EYE_CLOSED_THRESHOLD
                    and right_closure >= EYE_CLOSED_THRESHOLD
                )
                both_eyes_open = (
                    left_closure <= EYE_OPEN_THRESHOLD
                    and right_closure <= EYE_OPEN_THRESHOLD
                )

                if both_eyes_open and not driver_not_visible and alarm_active:
                    alarm_active = False
                    alarm_triggered_at = None
                    next_beep_at = None
                    if alarm_channel is not None:
                        alarm_channel.stop()
                    alarm_channel = None

                if not eyes_are_closing and both_eyes_closed:
                    eyes_are_closing = True
                    closure_started_at = now
                    current_closure_duration = 0.0
                elif eyes_are_closing and both_eyes_open:
                    eyes_are_closing = False
                    closure_started_at = None
                    current_closure_duration = 0.0

                if eyes_are_closing and closure_started_at is not None:
                    current_closure_duration = now - closure_started_at

                eye_indices = [
                    33, 7, 163, 144, 145, 153, 154, 155,
                    133, 173, 157, 158, 159, 160, 161, 246,
                    362, 382, 381, 380, 374, 373, 390, 249,
                    263, 466, 388, 387, 386, 385, 384, 398
                ]
                landmarks = result.face_landmarks[0]
                x_values = [int(landmarks[i].x * width) for i in eye_indices]
                y_values = [int(landmarks[i].y * height) for i in eye_indices]
                padding = 25
                x_min = max(0, min(x_values) - padding)
                x_max = min(width, max(x_values) + padding)
                y_min = max(0, min(y_values) - padding)
                y_max = min(height, max(y_values) + padding)

                if x_max > x_min and y_max > y_min:
                    eye_region = frame[y_min:y_max, x_min:x_max]
                    if eye_region.size > 0:
                        eye_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
                        eye_resized = cv2.resize(eye_rgb, (160, 68))
                        eye_crop = pygame.image.frombytes(
                            eye_resized.tobytes(), (160, 68), "RGB"
                        )
            else:
                if not driver_not_visible and not alarm_active:
                    eyes_are_closing = False
                    closure_started_at = None
                    current_closure_duration = 0.0

            show_red_warning = (
                (eyes_are_closing and current_closure_duration > DROWSINESS_WARNING_LIMIT)
                or driver_not_visible
            )

            was_alarm_active = alarm_active
            is_drowsy_critical = (
                analysis_available
                and eyes_are_closing
                and closure_started_at is not None
                and current_closure_duration > ALARM_CLOSURE_LIMIT
            )

            if is_drowsy_critical or driver_not_visible:
                alarm_active = True

            if alarm_active and not was_alarm_active:
                alarm_triggered_at = now

            if alarm_active:
                if next_beep_at is None or now >= next_beep_at:
                    if alarm_sound is not None:
                        alarm_channel = alarm_sound.play()
                    time_active = (now - alarm_triggered_at) if alarm_triggered_at else 0.0
                    interval = get_beep_interval(time_active)
                    next_beep_at = now + interval
            else:
                next_beep_at = None
                if alarm_channel is not None:
                    alarm_channel.stop()
                alarm_channel = None

            if driver_not_visible:
                status = "CRITICAL: DRIVER NOT VISIBLE!"
                status_color = (255, 85, 85)
            elif alarm_active:
                status = "CRITICAL: ACUTE DROWSINESS DETECTED!"
                status_color = (255, 85, 85)
            elif show_red_warning:
                status = "WARNING: DROWSINESS DETECTED"
                status_color = (255, 200, 70)
            else:
                status = "HIGHWAY MONITORING ACTIVE"
                status_color = (80, 255, 140)

            keys = pygame.key.get_pressed()

            # NITHO Drive Pro ONE Wheel & Pedal Input Mapping
            wheel_steer_axis = 0.0
            wheel_throttle = False
            wheel_brake = False

            if wheel is not None:
                if wheel.get_numaxes() > 0:
                    axis_val = wheel.get_axis(0) # Axis 0: Steering
                    if abs(axis_val) > 0.08:
                        wheel_steer_axis = -axis_val
                    
                    if wheel.get_numaxes() > 4:
                        accel_axis = wheel.get_axis(4)
                        if accel_axis > -0.5:
                            wheel_throttle = True

                    if wheel.get_numaxes() > 3:
                        brake_axis = wheel.get_axis(3)
                        if brake_axis > -0.5:
                            wheel_brake = True

            if not alarm_active:
                if keys[K_UP] or keys[ord('w')] or wheel_throttle:
                    is_braking = False
                    is_slow_pressed = keys[K_SPACE] or keys[ord('s')]
                    target_max = slow_displayed_kph if is_slow_pressed else max_displayed_kph
                    displayed_speed_kph = min(target_max, displayed_speed_kph + 1.1)
                elif is_braking or wheel_brake:
                    if wheel_brake and not is_braking:
                        is_braking = True
                        braking_start_speed = displayed_speed_kph
                        braking_start_time = time.monotonic()
                    brake_elapsed = now - braking_start_time
                    displayed_speed_kph = max(0.0, braking_start_speed - (10.0 * brake_elapsed))
                    if displayed_speed_kph <= 0.0:
                        displayed_speed_kph = 0.0
                        is_braking = False
                else:
                    is_slow_pressed = keys[K_SPACE] or keys[ord('s')]
                    if is_slow_pressed:
                        if displayed_speed_kph > slow_displayed_kph:
                            displayed_speed_kph = max(slow_displayed_kph, displayed_speed_kph - 3.0)
                        else:
                            displayed_speed_kph = slow_displayed_kph
                    else:
                        displayed_speed_kph = max(min_displayed_kph, displayed_speed_kph - (2.0 / TARGET_FPS))

                displayed_speed_kph = min(displayed_speed_kph, max_displayed_kph)

                # Manual Lane Retention & Steering Wheel Support (No Auto-Centering)
                max_tilt = 10.0
                steer_step = 0.03
                if keys[K_LEFT] or keys[ord('a')]:
                    target_player_x = max(-0.85, target_player_x - steer_step)
                    wheel_angle = max(-max_tilt, wheel_angle - 2.5)
                elif keys[K_RIGHT] or keys[ord('d')]:
                    target_player_x = min(0.85, target_player_x + steer_step)
                    wheel_angle = min(max_tilt, wheel_angle + 2.5)
                elif wheel_steer_axis != 0.0:
                    target_player_x = max(-0.85, min(0.85, target_player_x + wheel_steer_axis * steer_step))
                    wheel_angle = max(-max_tilt, min(max_tilt, wheel_steer_axis * max_tilt))
                else:
                    # Maintain target_player_x position; only relax the visual wheel rotation angle
                    wheel_angle *= 0.8
            else:
                is_braking = False
                # Smoothly decelerate at 20 km/hr^2 and steer onto the side shoulder to park
                displayed_speed_kph = max(0.0, displayed_speed_kph - (20.0 / TARGET_FPS))
                target_player_x = -0.78  # Pulls vehicle over to the right-hand side of the road
                if displayed_speed_kph > 0.5 and abs(player_x - target_player_x) > 0.05:
                    wheel_angle = min(10.0, wheel_angle + 1.5)
                else:
                    wheel_angle *= 0.8

            actual_speed_kph = displayed_speed_kph * ACTUAL_SPEED_RATIO
            actual_world_velocity = actual_speed_kph / SPEED_SCALE

            if not alarm_active:
                game_score += actual_world_velocity * 10.0

            steer_factor = 0.08 if alarm_active else 0.15
            if displayed_speed_kph > 0.1 or not alarm_active:
                player_x += (target_player_x - player_x) * steer_factor
            player_x = max(-0.85, min(0.85, player_x))
            world_z += actual_world_velocity * 18.0

            # ==========================================================
            # 5. RENDER EVERYTHING ONTO THE HIGH-RES CANVAS
            # ==========================================================
            canvas.blit(sky_surface, (0, 0))

            def get_highway_curve(z_pos):
                return math.sin(z_pos * 0.012) * 2.2 + math.cos(z_pos * 0.006) * 1.4

            parallax_x = int(-get_highway_curve(world_z) * 18)

            far_peaks = [
                (-100, horizon_y), (40, 125), (190, 165), (340, 105),
                (510, 160), (670, 118), (820, 155), (1020, horizon_y)
            ]
            far_poly = [(x + parallax_x // 2, y) for x, y in far_peaks]
            pygame.draw.polygon(canvas, (55, 72, 102), far_poly)

            near_peaks = [
                (-100, horizon_y), (90, 155), (250, 180), (430, 140),
                (610, 175), (770, 148), (1020, horizon_y)
            ]
            near_poly = [(x + parallax_x, y) for x, y in near_peaks]
            pygame.draw.polygon(canvas, (42, 88, 72), near_poly)

            pygame.draw.rect(canvas, (38, 115, 52), (0, horizon_y, WINDOW_WIDTH, WINDOW_HEIGHT - horizon_y))

            draw_distance = 135
            segment_length = 1.0
            player_segment_index = int(world_z / segment_length)
            base_curve = get_highway_curve(world_z)
            
            render_segments = []
            for i in range(draw_distance, -1, -1):
                seg_idx = player_segment_index + i
                seg_z = (seg_idx * segment_length) - world_z
                if seg_z <= 0.1:
                    seg_z = 0.1

                curve_center = get_highway_curve(world_z + seg_z) - base_curve
                
                scale = 1.0 / seg_z
                screen_x = WINDOW_WIDTH // 2 + int((player_x - curve_center) * 750 * scale)
                screen_y = horizon_y + int(240 * scale)
                road_width = int(1120 * scale)

                render_segments.append({
                    "z": seg_z,
                    "x": screen_x,
                    "y": screen_y,
                    "w": road_width,
                    "scale": scale,
                    "idx": seg_idx,
                    "curve": curve_center
                })

            for n in range(len(render_segments) - 1):
                s1 = render_segments[n]
                s2 = render_segments[n + 1]

                if s2["y"] <= s1["y"]:
                    continue

                band_even = ((s1["idx"] // 3) % 2 == 0)
                grass_color = (46, 132, 58) if band_even else (38, 115, 50)
                pygame.draw.rect(canvas, grass_color, (0, s1["y"], WINDOW_WIDTH, s2["y"] - s1["y"]))

                kerb_w1 = max(2, int(s1["w"] * 0.08))
                kerb_w2 = max(2, int(s2["w"] * 0.08))
                kerb_color = (220, 45, 45) if ((s1["idx"] // 2) % 2 == 0) else (240, 240, 245)
                kerb_poly = [
                    (s1["x"] - s1["w"] // 2 - kerb_w1, s1["y"]),
                    (s1["x"] + s1["w"] // 2 + kerb_w1, s1["y"]),
                    (s2["x"] + s2["w"] // 2 + kerb_w2, s2["y"]),
                    (s2["x"] - s2["w"] // 2 - kerb_w2, s2["y"]),
                ]
                pygame.draw.polygon(canvas, kerb_color, kerb_poly)

                fog = min(1.0, s1["z"] / draw_distance)
                base_asphalt = 54 if band_even else 62
                r_c = int(base_asphalt * (1 - fog * 0.35) + 90 * (fog * 0.35))
                g_c = int(base_asphalt * (1 - fog * 0.35) + 95 * (fog * 0.35))
                b_c = int((base_asphalt + 6) * (1 - fog * 0.35) + 110 * (fog * 0.35))

                road_poly = [
                    (s1["x"] - s1["w"] // 2, s1["y"]),
                    (s1["x"] + s1["w"] // 2, s1["y"]),
                    (s2["x"] + s2["w"] // 2, s2["y"]),
                    (s2["x"] - s2["w"] // 2, s2["y"]),
                ]
                pygame.draw.polygon(canvas, (r_c, g_c, b_c), road_poly)

                line_w = max(1, int(4 * s1["scale"]))
                pygame.draw.line(canvas, (230, 235, 240), (s1["x"] - s1["w"] // 2, s1["y"]), (s2["x"] - s2["w"] // 2, s2["y"]), line_w)
                pygame.draw.line(canvas, (230, 235, 240), (s1["x"] + s1["w"] // 2, s1["y"]), (s2["x"] + s2["w"] // 2, s2["y"]), line_w)

                center_w = max(1, int(3 * s1["scale"]))
                pygame.draw.line(canvas, (235, 195, 45), (s1["x"], s1["y"]), (s2["x"], s2["y"]), center_w)

                if (s1["idx"] // 3) % 2 == 0:
                    for lane_offset in (-0.28, 0.28):
                        lx1 = s1["x"] + int(s1["w"] * lane_offset)
                        lx2 = s2["x"] + int(s2["w"] * lane_offset)
                        pygame.draw.line(canvas, (215, 220, 225), (lx1, s1["y"]), (lx2, s2["y"]), max(1, int(3 * s1["scale"])))

                idx = s1["idx"]
                if idx % 9 == 0:
                    tree_x = s1["x"] - int(s1["w"] * 0.82)
                    draw_scenery_object(canvas, "tree", tree_x, s1["y"], s1["scale"])
                if idx % 11 == 0:
                    tree_x = s1["x"] + int(s1["w"] * 0.86)
                    draw_scenery_object(canvas, "tree", tree_x, s1["y"], s1["scale"])
                if idx % 16 == 0:
                    lamp_x = s1["x"] - int(s1["w"] * 0.62)
                    draw_scenery_object(canvas, "lamp", lamp_x, s1["y"], s1["scale"])
                elif idx % 16 == 8:
                    lamp_x = s1["x"] + int(s1["w"] * 0.62)
                    draw_scenery_object(canvas, "lamp", lamp_x, s1["y"], s1["scale"])

            pygame.draw.polygon(canvas, (28, 32, 42), [(0, 0), (140, 0), (35, WINDOW_HEIGHT), (0, WINDOW_HEIGHT)])
            pygame.draw.polygon(canvas, (42, 48, 62), [(120, 0), (142, 0), (38, WINDOW_HEIGHT), (25, WINDOW_HEIGHT)])
            pygame.draw.polygon(canvas, (28, 32, 42), [(WINDOW_WIDTH, 0), (WINDOW_WIDTH - 140, 0), (WINDOW_WIDTH - 35, WINDOW_HEIGHT), (WINDOW_WIDTH, WINDOW_HEIGHT)])
            pygame.draw.polygon(canvas, (42, 48, 62), [(WINDOW_WIDTH - 120, 0), (WINDOW_WIDTH - 142, 0), (WINDOW_WIDTH - 38, WINDOW_HEIGHT), (WINDOW_WIDTH - 25, WINDOW_HEIGHT)])

            hood_shift = int(wheel_angle * 1.5)
            hood_top_y = WINDOW_HEIGHT - 178
            hood_poly = [
                (WINDOW_WIDTH // 2 - 250, WINDOW_HEIGHT),
                (WINDOW_WIDTH // 2 - 175 + hood_shift, hood_top_y),
                (WINDOW_WIDTH // 2 + 175 + hood_shift, hood_top_y),
                (WINDOW_WIDTH // 2 + 250, WINDOW_HEIGHT)
            ]
            pygame.draw.polygon(canvas, (25, 110, 215), hood_poly)
            pygame.draw.line(canvas, (75, 165, 255), (WINDOW_WIDTH // 2 - 110 + hood_shift, hood_top_y), (WINDOW_WIDTH // 2 - 160, WINDOW_HEIGHT), 3)
            pygame.draw.line(canvas, (75, 165, 255), (WINDOW_WIDTH // 2 + 110 + hood_shift, hood_top_y), (WINDOW_WIDTH // 2 + 160, WINDOW_HEIGHT), 3)
            pygame.draw.line(canvas, (15, 75, 160), (WINDOW_WIDTH // 2 + hood_shift, hood_top_y), (WINDOW_WIDTH // 2, WINDOW_HEIGHT), 4)

            dashboard_y = WINDOW_HEIGHT - 155
            dash_poly = [
                (0, dashboard_y + 15),
                (180, dashboard_y),
                (WINDOW_WIDTH - 180, dashboard_y),
                (WINDOW_WIDTH, dashboard_y + 15),
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                (0, WINDOW_HEIGHT)
            ]
            pygame.draw.polygon(canvas, (26, 30, 38), dash_poly)
            pygame.draw.lines(canvas, (65, 75, 92), False, dash_poly[:4], 5)

            cluster_rect = pygame.Rect(WINDOW_WIDTH // 2 - 155, dashboard_y - 22, 310, 90)
            pygame.draw.ellipse(canvas, (16, 20, 28), cluster_rect)
            pygame.draw.ellipse(canvas, (55, 65, 80), cluster_rect, 4)
            
            hazard_center = (WINDOW_WIDTH // 2, dashboard_y + 24)
            is_hazard_lit = alarm_active and (int(now * 8) % 2 == 0)
            outer_haz = (240, 40, 40) if is_hazard_lit else (75, 30, 30)
            pygame.draw.circle(canvas, outer_haz, hazard_center, 15)
            pygame.draw.circle(canvas, (180, 190, 205), hazard_center, 15, 2)
            tri_pts = [
                (hazard_center[0], hazard_center[1] - 8),
                (hazard_center[0] - 8, hazard_center[1] + 6),
                (hazard_center[0] + 8, hazard_center[1] + 6)
            ]
            pygame.draw.polygon(canvas, (255, 230, 90) if is_hazard_lit else (160, 160, 160), tri_pts, 2)

            wheel_center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 38)
            wheel_radius = 122
            rad = math.radians(wheel_angle)

            for spoke_deg, thickness in [(-90, 14), (90, 14), (180, 16)]:
                s_rad = rad + math.radians(spoke_deg)
                sx = wheel_center[0] + int((wheel_radius - 10) * math.sin(s_rad))
                sy = wheel_center[1] - int((wheel_radius - 10) * math.cos(s_rad))
                pygame.draw.line(canvas, (48, 54, 66), wheel_center, (sx, sy), thickness)

            pygame.draw.circle(canvas, (20, 23, 30), wheel_center, wheel_radius, 24)
            pygame.draw.circle(canvas, (55, 62, 75), wheel_center, wheel_radius - 2, 3)

            top_x = wheel_center[0] + int((wheel_radius - 12) * math.sin(rad))
            top_y = wheel_center[1] - int((wheel_radius - 12) * math.cos(rad))
            pygame.draw.circle(canvas, (235, 45, 45), (top_x, top_y), 9)

            pygame.draw.circle(canvas, (35, 40, 52), wheel_center, 34)
            pygame.draw.circle(canvas, (90, 100, 120), wheel_center, 34, 3)
            pygame.draw.circle(canvas, (25, 110, 215), wheel_center, 12)

            mirror_rect = pygame.Rect(WINDOW_WIDTH // 2 - 210, 8, 420, 38)
            pygame.draw.rect(canvas, (18, 22, 30), mirror_rect, border_radius=12)
            pygame.draw.rect(canvas, status_color, mirror_rect, 2, border_radius=12)
            title_surface = alert_font.render(status, True, status_color)
            canvas.blit(title_surface, title_surface.get_rect(center=mirror_rect.center))

            telemetry_bg = pygame.Surface((155, 78), pygame.SRCALPHA)
            pygame.draw.rect(telemetry_bg, (15, 22, 32, 190), (0, 0, 155, 78), border_radius=10)
            pygame.draw.rect(telemetry_bg, (80, 150, 220, 200), (0, 0, 155, 78), 2, border_radius=10)
            canvas.blit(telemetry_bg, (18, 12))

            lap_surf = hud_font.render("LAP 2/3", True, (240, 248, 255))
            pos_surf = hud_font.render("POS: 4/8", True, (120, 210, 255))
            canvas.blit(lap_surf, (32, 18))
            canvas.blit(pos_surf, (32, 48))

            elapsed_game_time = int(now - start_time)
            mins = elapsed_game_time // 60
            secs = elapsed_game_time % 60
            msecs = int((now % 1) * 100)

            right_hud_bg = pygame.Surface((146, 128), pygame.SRCALPHA)
            pygame.draw.rect(right_hud_bg, (15, 22, 32, 190), (0, 0, 146, 128), border_radius=10)
            pygame.draw.rect(right_hud_bg, (80, 150, 220, 200), (0, 0, 146, 128), 2, border_radius=10)
            canvas.blit(right_hud_bg, (WINDOW_WIDTH - 164, 12))

            timer_surf = hud_font.render(f"{mins:02d}:{secs:02d}:{msecs:02d}", True, (255, 255, 255))
            canvas.blit(timer_surf, (WINDOW_WIDTH - 148, 17))

            track_rect = pygame.Rect(WINDOW_WIDTH - 146, 54, 110, 68)
            pygame.draw.ellipse(canvas, (90, 110, 135), track_rect, 5)
            pygame.draw.ellipse(canvas, (210, 225, 240), track_rect, 2)
            dot_angle = world_z * 0.008
            dot_x = track_rect.centerx + int((track_rect.width // 2 - 2) * math.cos(dot_angle))
            dot_y = track_rect.centery + int((track_rect.height // 2 - 2) * math.sin(dot_angle))
            pygame.draw.circle(canvas, (255, 215, 0), (dot_x, dot_y), 6)
            pygame.draw.circle(canvas, (255, 255, 255), (dot_x, dot_y), 6, 2)

            cam_x, cam_y = 24, WINDOW_HEIGHT - 122
            pygame.draw.rect(canvas, (14, 18, 26), (cam_x - 4, cam_y - 22, 168, 94), border_radius=6)
            frame_col = (255, 70, 70) if show_red_warning else (65, 220, 130)
            pygame.draw.rect(canvas, frame_col, (cam_x - 4, cam_y - 22, 168, 94), 2, border_radius=6)
            cam_lbl = game_font.render("BIOMETRIC EYE TRACKER", True, frame_col)
            canvas.blit(cam_lbl, (cam_x + 4, cam_y - 19))

            if eye_crop is not None:
                canvas.blit(eye_crop, (cam_x, cam_y))
                pygame.draw.line(canvas, frame_col, (cam_x + 80, cam_y + 8), (cam_x + 80, cam_y + 60), 1)
                pygame.draw.line(canvas, frame_col, (cam_x + 15, cam_y + 34), (cam_x + 145, cam_y + 34), 1)
            else:
                no_sig = game_font.render("NO FACE SIGNAL", True, (255, 90, 90))
                canvas.blit(no_sig, no_sig.get_rect(center=(cam_x + 80, cam_y + 34)))

            draw_speed_limit_sign(
                canvas,
                center=(WINDOW_WIDTH - 205, WINDOW_HEIGHT - 85),
                limit_value=SPEED_LIMIT_KPH,
                font=hud_font,
                sub_font=game_font
            )

            draw_analog_gauge(
                canvas,
                center=(WINDOW_WIDTH - 92, WINDOW_HEIGHT - 78),
                radius=68,
                speed_kph=displayed_speed_kph,
                max_kph=100,
                hud_font=hud_font,
                small_font=game_font
            )

            canvas.blit(vignette_surf, (0, 0))

            if show_red_warning:
                pulse_alpha = int(85 + 45 * math.sin(now * 10))
                overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                overlay.fill((255, 0, 0, pulse_alpha))
                canvas.blit(overlay, (0, 0))

                banner_rect = pygame.Rect(WINDOW_WIDTH // 2 - 290, WINDOW_HEIGHT // 2 - 75, 580, 70)
                pygame.draw.rect(canvas, (20, 0, 0), banner_rect, border_radius=12)
                pygame.draw.rect(canvas, (255, 70, 70), banner_rect, 4, border_radius=12)

                msg = "DRIVER NOT VISIBLE!" if driver_not_visible else "EXPECTED DROWSINESS"
                warning_txt = warning_overlay_font.render(msg, True, (255, 255, 255))
                canvas.blit(warning_txt, warning_txt.get_rect(center=banner_rect.center))

            scaled_surface = pygame.transform.smoothscale(canvas.surface, (monitor_width, monitor_height))
            screen.blit(scaled_surface, (0, 0))

            pygame.display.flip()
            clock.tick(TARGET_FPS)

    finally:
        pygame.draw = _ORIG_DRAW
        if alarm_channel is not None:
            alarm_channel.stop()
        if alarm_sound is not None:
            alarm_sound.stop()
        if camera is not None:
            camera.release()
        if landmarker is not None:
            landmarker.close()
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nProgram stopped:", error)
        try:
            input("Press Enter to close...")
        except EOFError:
            pass