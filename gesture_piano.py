"""用單手數字手勢選擇 C 大調級數和弦，並即時演奏分解和弦。"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame

from hand_gesture import CONNECTIONS, MODEL_PATH, download_model, recognize_number


ROOT = Path(__file__).resolve().parent
PIANO_DIR = ROOT / "assets" / "piano"
WINDOW_SIZE = (1160, 720)
CAMERA_BOX = (28, 128, 600, 450)
SCALE = [60, 62, 64, 65, 67, 69, 71]  # C4 大調音階
ROMAN = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
CHORD_NAMES = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
QUALITIES = ["MAJOR", "MINOR", "MINOR", "MAJOR", "MAJOR", "MINOR", "DIMINISHED"]
NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
FILE_NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
ARPEGGIO_STEPS = [0, 1, 2, 3, 2, 1]  # 根音、三音、五音、八度、五音、三音
GUITAR_RHYTHM = [True] * 8
GUITAR_BPM = 96

BG = (15, 18, 25)
PANEL = (25, 30, 40)
PANEL_LIGHT = (33, 40, 52)
TEXT = (239, 239, 232)
MUTED = (151, 163, 177)
GOLD = (237, 189, 115)
GOLD_DARK = (111, 78, 43)
MINT = (133, 221, 185)


def sample_path(midi_note: int) -> Path:
    octave = midi_note // 12 - 1
    name = FILE_NOTE_NAMES[midi_note % 12]
    return PIANO_DIR / f"{name}{octave}.ogg"


def load_piano_samples() -> dict[int, pygame.mixer.Sound]:
    samples: dict[int, pygame.mixer.Sound] = {}
    for midi_note in range(48, 84):  # C3 到 B5
        path = sample_path(midi_note)
        if not path.exists():
            raise FileNotFoundError(f"缺少鋼琴音色樣本：{path}")
        sound = pygame.mixer.Sound(str(path))
        # 加一層柔和飽和增益，提高取樣音的平均響度，同時控制尖銳的鋼琴起音。
        pcm = pygame.sndarray.array(sound).astype(np.float32) / 32768.0
        pcm = np.tanh(pcm * 2.0) / np.tanh(2.0)
        samples[midi_note] = pygame.sndarray.make_sound((pcm * 32767.0).astype(np.int16))
    return samples


def note_name_to_midi(note_name: str) -> int:
    match = re.fullmatch(r"([A-G])(#?)(\d+)", note_name)
    if not match:
        raise ValueError(f"無法解析吉他音符：{note_name}")
    pitch_classes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    pitch = pitch_classes[match.group(1)] + (1 if match.group(2) else 0)
    return (int(match.group(3)) + 1) * 12 + pitch


def pitch_shift(pcm: np.ndarray, semitones: int) -> np.ndarray:
    ratio = 2.0 ** (semitones / 12.0)
    new_length = max(2, round(len(pcm) / ratio))
    positions = np.arange(new_length, dtype=np.float32) * ratio
    positions = np.minimum(positions, len(pcm) - 1)
    old_positions = np.arange(len(pcm), dtype=np.float32)
    channels = [np.interp(positions, old_positions, pcm[:, channel]) for channel in range(pcm.shape[1])]
    return np.stack(channels, axis=1)


def load_guitar_samples() -> dict[int, pygame.mixer.Sound]:
    source_samples: dict[int, np.ndarray] = {}
    for path in (ROOT / "assets" / "electric_guitar").glob("*.flac"):
        source_midi = note_name_to_midi(path.stem.split("_", 1)[0])
        source_sound = pygame.mixer.Sound(str(path))
        pcm = pygame.sndarray.array(source_sound).astype(np.float32) / 32768.0
        # 明顯的音箱過載，做出電吉他破音音色。
        driven = np.tanh(pcm * 3.8) / np.tanh(3.8)
        source_samples[source_midi] = (driven * 32767.0).astype(np.int16)

    if not source_samples:
        raise FileNotFoundError("找不到電吉他音色樣本：assets/electric_guitar/*.flac")

    samples: dict[int, pygame.mixer.Sound] = {}
    for target_midi in range(48, 84):
        source_midi = min(source_samples, key=lambda pitch: abs(pitch - target_midi))
        shifted = pitch_shift(source_samples[source_midi], target_midi - source_midi)
        samples[target_midi] = pygame.sndarray.make_sound(np.clip(shifted, -32768, 32767).astype(np.int16))
    return samples


def make_power_chord(root: int, samples: dict[int, pygame.mixer.Sound]) -> pygame.mixer.Sound:
    notes = [root, root + 7, root + 12]
    offsets = [0, 480, 960]  # 10 ms 間隔的撥弦
    pcm_notes = [pygame.sndarray.array(samples[note]).astype(np.float32) for note in notes]
    total_length = max(offset + len(pcm) for offset, pcm in zip(offsets, pcm_notes))
    mix = np.zeros((total_length, 2), dtype=np.float32)
    for offset, pcm, gain in zip(offsets, pcm_notes, (0.72, 0.60, 0.50)):
        mix[offset:offset + len(pcm)] += pcm * gain
    normalized = mix / 32768.0
    limited = np.tanh(normalized * 1.12) / np.tanh(1.12)
    return pygame.sndarray.make_sound((limited * 32767.0).astype(np.int16))


def chord_for_degree(degree: int) -> list[int]:
    start = degree - 1
    return [SCALE[index % 7] + 12 * (index // 7) for index in (start, start + 2, start + 4)]


def power_root_for_degree(degree: int) -> int:
    return SCALE[degree - 1]


def play_note(samples: dict[int, pygame.mixer.Sound], midi_note: int, volume: float, pan: float = 0.5) -> None:
    sound = samples[midi_note]
    channel = sound.play()
    if channel:
        # 輕微左右交替，保留鋼琴立體感，不讓聲像大幅移動。
        left = volume * (1.0 - max(0.0, min(1.0, pan)) * 0.24)
        right = volume * (0.76 + max(0.0, min(1.0, pan)) * 0.24)
        channel.set_volume(left, right)


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        font_dir / ("msjhbd.ttc" if bold else "msjh.ttc"),
        font_dir / "mingliu.ttc",
    ]
    for path in candidates:
        if path.exists():
            return pygame.font.Font(str(path), size)
    return pygame.font.SysFont("segoeui", size, bold=bold)


def draw_text(screen, font, text: str, color, position, anchor="topleft") -> None:
    surface = font.render(text, True, color)
    rect = surface.get_rect(**{anchor: position})
    screen.blit(surface, rect)


def draw_rounded_rect(screen, color, rect, radius=18, border=0, border_color=None) -> None:
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(screen, border_color, rect, width=border, border_radius=radius)


def draw_hand_overlay(frame, points) -> None:
    height, width = frame.shape[:2]
    for start, end in CONNECTIONS:
        a, b = points[start], points[end]
        cv2.line(frame, (int(a.x * width), int(a.y * height)),
                 (int(b.x * width), int(b.y * height)), (125, 218, 185), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, (int(point.x * width), int(point.y * height)), 4, (245, 245, 238), -1, cv2.LINE_AA)


def camera_surface(frame, box):
    x, y, w, h = box
    preview = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB").convert()


def draw_keyboard(screen, x: int, y: int, width: int, active_chord: list[int] | None, fonts) -> None:
    small_font, tiny_font = fonts
    white_notes = [m for m in range(60, 84) if m % 12 in (0, 2, 4, 5, 7, 9, 11)]
    key_w = width / len(white_notes)
    key_h = 80
    chord_notes = set(active_chord or [])
    chord_pitch_classes = {note % 12 for note in chord_notes}

    for index, midi_note in enumerate(white_notes):
        key_x = round(x + index * key_w)
        next_x = round(x + (index + 1) * key_w)
        color = GOLD if midi_note % 12 in chord_pitch_classes else (226, 228, 226)
        pygame.draw.rect(screen, color, (key_x, y, next_x - key_x, key_h), border_radius=3)
        pygame.draw.line(screen, (76, 82, 89), (key_x, y + 2), (key_x, y + key_h - 2), 1)
        if midi_note % 12 == 0:
            draw_text(screen, tiny_font, f"C{midi_note // 12 - 1}", (78, 84, 91), (key_x + 4, y + key_h - 20))

    white_before = 0
    black_w = key_w * 0.60
    black_h = key_h * 0.62
    for midi_note in range(60, 84):
        pitch = midi_note % 12
        if pitch in (1, 3, 6, 8, 10):
            key_x = round(x + white_before * key_w - black_w / 2)
            color = GOLD_DARK if pitch in chord_pitch_classes else (36, 40, 48)
            pygame.draw.rect(screen, color, (key_x, y, round(black_w), black_h), border_radius=3)
            pygame.draw.rect(screen, (12, 14, 18), (key_x, y, round(black_w), black_h), width=1, border_radius=3)
        elif pitch in (0, 2, 4, 5, 7, 9, 11):
            white_before += 1


def main() -> None:
    download_model()
    pygame.mixer.pre_init(frequency=48000, size=-16, channels=2, buffer=512)
    pygame.init()
    pygame.display.set_caption("Gesture Piano · 手勢鋼琴")
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    clock = pygame.time.Clock()
    pygame.mixer.set_num_channels(32)
    piano_samples = load_piano_samples()
    guitar_samples = load_guitar_samples()
    power_chords = {
        degree: make_power_chord(power_root_for_degree(degree), guitar_samples)
        for degree in range(1, 8)
    }

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    if not camera.isOpened():
        pygame.quit()
        raise RuntimeError("無法開啟相機。請確認相機已連接，且沒有被其他程式占用。")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.5,
    )

    title_font = get_font(30, True)
    heading_font = get_font(15, True)
    body_font = get_font(21)
    small_font = get_font(16)
    tiny_font = get_font(12)
    chord_font = get_font(32, True)
    number_font = get_font(84, True)
    camera_frame = None
    active_degree = None
    candidate_degree = None
    candidate_frames = 0
    last_valid_gesture = 0.0
    arpeggio_step = 0
    next_note_at = 0.0
    keyboard_override_until = 0.0
    instrument = "piano"
    guitar_channels: list[tuple[pygame.mixer.Channel, float]] = []
    started_at = time.monotonic()
    running = True

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_TAB, pygame.K_p, pygame.K_g):
                        new_instrument = "guitar" if event.key == pygame.K_g else "piano" if event.key == pygame.K_p else ("guitar" if instrument == "piano" else "piano")
                        if new_instrument != instrument:
                            instrument = new_instrument
                            arpeggio_step = 0
                            next_note_at = time.monotonic()
                            pygame.mixer.fadeout(100)
                            guitar_channels.clear()
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        window_w = screen.get_width()
                        right_x = 668
                        right_w = max(450, window_w - right_x - 24)
                        switch_rect = pygame.Rect(right_x + right_w - 158, 103, 140, 34)
                        if switch_rect.collidepoint(event.pos):
                            instrument = "guitar" if instrument == "piano" else "piano"
                            arpeggio_step = 0
                            next_note_at = time.monotonic()
                            pygame.mixer.fadeout(100)
                            guitar_channels.clear()
                    elif event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_7:
                        active_degree = event.key - pygame.K_0
                        arpeggio_step = 0
                        next_note_at = time.monotonic()
                        keyboard_override_until = next_note_at + 5.0
                    elif event.type == pygame.VIDEORESIZE:
                        screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

                ok, frame = camera.read()
                raw_number = None
                if ok:
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int((time.monotonic() - started_at) * 1000)
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    if result.hand_landmarks:
                        points = result.hand_landmarks[0]
                        raw_number = recognize_number(points)
                        draw_hand_overlay(frame, points)
                    camera_frame = frame

                now = time.monotonic()
                if raw_number is not None and 1 <= raw_number <= 7:
                    keyboard_override_until = 0.0
                    last_valid_gesture = now
                    if raw_number == candidate_degree:
                        candidate_frames += 1
                    else:
                        candidate_degree = raw_number
                        candidate_frames = 1
                    if candidate_frames >= 4 and active_degree != raw_number:
                        active_degree = raw_number
                        arpeggio_step = 0
                        next_note_at = now
                else:
                    candidate_degree = None
                    candidate_frames = 0
                    if active_degree is not None and now - last_valid_gesture > 0.65 and now >= keyboard_override_until:
                        active_degree = None

                current_chord = chord_for_degree(active_degree) if active_degree else None
                current_guitar_chord = (
                    [power_root_for_degree(active_degree), power_root_for_degree(active_degree) + 7,
                     power_root_for_degree(active_degree) + 12]
                    if active_degree else None
                )
                display_chord = current_chord if instrument == "piano" else current_guitar_chord

                remaining_guitar_channels = []
                for channel, fade_at in guitar_channels:
                    if now >= fade_at and channel.get_busy():
                        channel.fadeout(150)
                    elif now < fade_at and channel.get_busy():
                        remaining_guitar_channels.append((channel, fade_at))
                guitar_channels = remaining_guitar_channels

                if active_degree is not None and now >= next_note_at:
                    if instrument == "piano":
                        pattern = [current_chord[0], current_chord[1], current_chord[2], current_chord[0] + 12, current_chord[2], current_chord[1]]
                        note = pattern[arpeggio_step]
                        pan = (0.35, 0.68, 0.43, 0.60, 0.72, 0.38)[arpeggio_step]
                        play_note(piano_samples, note, 1.0, pan)
                        if arpeggio_step == 0:
                            play_note(piano_samples, current_chord[0] - 12, 0.92, 0.5)
                        arpeggio_step = (arpeggio_step + 1) % len(pattern)
                    else:
                        if GUITAR_RHYTHM[arpeggio_step]:
                            channel = power_chords[active_degree].play()
                            if channel:
                                accent = 1.0 if arpeggio_step % 2 == 0 else 0.78
                                channel.set_volume(accent, accent)
                                guitar_channels.append((channel, now + 0.52))
                        arpeggio_step = (arpeggio_step + 1) % len(GUITAR_RHYTHM)
                    tempo = GUITAR_BPM if instrument == "guitar" else 112
                    next_note_at = now + 60 / tempo / 2

                window_w, window_h = screen.get_size()
                screen.fill(BG)
                draw_text(screen, title_font, "手勢鋼琴", TEXT, (30, 26))
                mode_name = "鋼琴分解和弦" if instrument == "piano" else "電吉他 Power Chord"
                draw_text(screen, heading_font, f"GESTURE PIANO  /  {mode_name}  /  C MAJOR", MUTED, (232, 43))
                draw_text(screen, small_font, "比出 1–7 選級數  ·  Tab 切換音色  ·  Q 離開", MUTED, (window_w - 28, 43), "topright")

                left_box = pygame.Rect(24, 92, 620, 536)
                draw_rounded_rect(screen, PANEL, left_box, 20)
                draw_text(screen, heading_font, "CAMERA  ·  手勢輸入", MUTED, (44, 108))
                if camera_frame is not None:
                    screen.blit(camera_surface(camera_frame, CAMERA_BOX), (38, 145))
                else:
                    draw_text(screen, body_font, "正在開啟相機…", MUTED, (240, 350))
                gesture_text = f"辨識手勢：{raw_number}" if raw_number is not None else "請將手掌朝向鏡頭"
                draw_text(screen, small_font, gesture_text, MINT if raw_number is not None else MUTED, (42, 598))

                right_x = 668
                right_w = max(450, window_w - right_x - 24)
                card = pygame.Rect(right_x, 92, right_w, 292)
                draw_rounded_rect(screen, PANEL, card, 20)
                draw_text(screen, heading_font, "目前音色  ·  CURRENT SOUND", MUTED, (right_x + 22, 110))
                switch_rect = pygame.Rect(right_x + right_w - 158, 103, 140, 34)
                draw_rounded_rect(screen, (73, 57, 40) if instrument == "guitar" else PANEL_LIGHT, switch_rect, 11, 1, GOLD if instrument == "guitar" else (58, 67, 80))
                switch_label = "切到電吉他" if instrument == "piano" else "切到鋼琴"
                draw_text(screen, tiny_font, switch_label, GOLD if instrument == "guitar" else TEXT, switch_rect.center, "center")
                if active_degree is None:
                    draw_text(screen, number_font, "—", (89, 101, 114), (right_x + 28, 130))
                    draw_text(screen, body_font, "保持一個 1–7 手勢開始演奏", TEXT, (right_x + 150, 164))
                    draw_text(screen, small_font, "每個級數會持續循環演奏分解和弦", MUTED, (right_x + 150, 200))
                else:
                    index = active_degree - 1
                    draw_text(screen, number_font, str(active_degree), GOLD, (right_x + 20, 130))
                    if instrument == "piano":
                        title = f"{ROMAN[index]}   {CHORD_NAMES[index]}"
                        quality = QUALITIES[index]
                        note_sequence = [current_chord[0], current_chord[1], current_chord[2], current_chord[0] + 12]
                        subhead = "ARPEGGIO  ·  分解和弦"
                    else:
                        title = f"{ROMAN[index]}   {NOTE_NAMES[power_root_for_degree(active_degree) % 12]}5"
                        quality = "ROOT  ·  FIFTH  ·  OCTAVE"
                        note_sequence = current_guitar_chord
                        subhead = "POWER CHORD  ·  根音五度八度"
                    draw_text(screen, chord_font, title, TEXT, (right_x + 154, 148))
                    draw_text(screen, body_font, f"C 大調第 {active_degree} 級  ·  {quality}", MUTED, (right_x + 156, 194))
                    arp_names = "  →  ".join(f"{NOTE_NAMES[n % 12]}{n // 12 - 1}" for n in note_sequence)
                    draw_text(screen, heading_font, subhead, GOLD, (right_x + 156, 246))
                    draw_text(screen, small_font, arp_names, TEXT, (right_x + 156, 271))

                draw_text(screen, heading_font, "SCALE DEGREES  ·  級數", MUTED, (right_x + 4, 398))
                row_y = 421
                row_gap = 23
                for index in range(7):
                    row = pygame.Rect(right_x, row_y + index * row_gap, right_w, 20)
                    selected = active_degree == index + 1
                    draw_rounded_rect(screen, (68, 55, 39) if selected else PANEL, row, 8)
                    accent = GOLD if selected else MUTED
                    draw_text(screen, small_font, f"{index + 1}   {ROMAN[index]}", accent, (right_x + 12, row_y + 1 + index * row_gap))
                    degree_chord = CHORD_NAMES[index] if instrument == "piano" else f"{NOTE_NAMES[SCALE[index] % 12]}5"
                    degree_quality = QUALITIES[index] if instrument == "piano" else "POWER"
                    draw_text(screen, small_font, degree_chord, TEXT if selected else MUTED, (right_x + 124, row_y + 1 + index * row_gap))
                    draw_text(screen, tiny_font, degree_quality, MUTED, (right_x + 230, row_y + 4 + index * row_gap))

                keyboard_y = min(608, window_h - 108)
                draw_text(screen, heading_font, "C4–B5  ·  和弦音高亮", MUTED, (right_x + 4, keyboard_y - 22))
                draw_keyboard(screen, right_x + 2, keyboard_y, right_w - 4, display_chord, (small_font, tiny_font))
                draw_text(screen, tiny_font, "Meta Piano · CC0 Electric Guitar · 數字鍵 1–7 可試彈", MUTED, (28, window_h - 22))
                pygame.display.flip()
                clock.tick(30)
    finally:
        camera.release()
        pygame.mixer.stop()
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, FileNotFoundError) as error:
        print(f"錯誤：{error}")
