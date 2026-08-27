

# BEAST - Milestone 1: Wake Word Detection ✅ COMPLETE

This is the first milestone of the BEAST project - a Windows background application with wake word detection.

**Status: COMPLETE (2026-08-22)** — "Hey Jarvis" wake word triggers reliably from real microphone input in the tray app.

## Features Implemented

1. Windows background application running in system tray
2. System tray icon with right-click menu:
   - Status (placeholder)
   - Settings (placeholder)
   - Activity Log (placeholder)
   - Emergency STOP
   - Quit
3. Wake-word detection using openWakeWord with pretrained `hey_jarvis_v0.1` model
4. Real microphone streaming via sounddevice at 16kHz, mono, float32, blocksize=1280
5. On wake-word detection: shows notification ("Beast is listening...") and logs the event
6. Configurable audio gain with clipping protection (`audio_gain` in settings.json)
7. Model-key mismatch fallback (uses max prediction value if configured key is absent)
8. Per-callback audio diagnostics (frames/min/max/RMS) and raw score logging on every inference
9. Basic configuration file (JSON)
10. Basic logging setup

## Dependencies

- Python 3.13+
- pystray
- openwakeword
- Pillow
- onnxruntime (installed as dependency of openwakeword)
- numpy, scipy, scikit-learn (installed as dependencies of openwakeword)

## Installation and Setup

1. **Clone or extract this repository** to `C:\Beast`

2. **Create and activate virtual environment**:
   ```bash
   cd C:\Beast\beast
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Activate virtual environment** (if not already active):
   ```bash
   cd C:\Beast\beast
   venv\Scripts\activate
   ```

2. **Run the application**:
   ```bash
   python app\main.py
   ```

3. **Test the application**:
   - Look for the Beast icon in your system tray (bottom-right corner)
   - Right-click the icon to see the menu
   - The application will show a notification every 30 seconds (simulated wake word detection)
   - Check the logs in `C:\Beast\logs\beast.log`

## Expected Behavior

When you run the application:
1. A Beast icon appears in the system tray
2. Right-clicking shows the menu with Status, Settings, Activity Log, Emergency STOP, and Quit
3. Every 30 seconds, you should see a Windows notification saying "Beast is listening..."
4. Each detection is logged to `C:\Beast\logs\beast.log` with timestamp
5. CPU usage should remain low (single-digit % when idle)

## Configuration

Configuration is stored in `C:\Beast\beast\config\settings.json`:
```json
{
  "microphone": null,
  "wake_sensitivity": 0.5,
  "wake_word": "hey jarvis",
  "wake_word_model": "hey_jarvis_v0.1",
  "model_path": null,
  "audio_sample_rate": 16000,
  "audio_chunk_size": 1024,
  "log_level": "INFO",
  "audio_gain": 12.0
}
```

## Notes on Wake Word Detection

- Uses the pretrained openWakeWord `hey_jarvis_v0.1` model (ONNX inference framework).
- Audio is streamed in 1280-sample blocks (80ms) to match openWakeWord's expected frame size exactly.
- `audio_gain` (currently 12.0x) compensates for a quiet microphone; signal is clipped to [-1, 1] before int16 conversion.
- If the model's prediction dict key doesn't match the configured key, the code falls back to the max prediction value (logged as KEY MISMATCH).
- The custom "Hey Beast" model training effort is archived in `beast/archive/custom_hey_beast_training/` (see its README for root-cause analysis and the V3 plan to resume later).

## Troubleshooting

1. **No icon in system tray**: Check if the application is running in the console
2. **Notifications not showing**: Ensure Windows notifications are enabled for the app
3. **Import errors**: Make sure virtual environment is activated and dependencies installed
4. **High CPU usage**: The wake word detection should be efficient; check logs for errors

## Next Steps (Milestone 2)

Milestone 1 is complete. Milestone 2 scope:
1. Add Speech-to-Text (faster-whisper)
2. Add local LLM (Qwen3-1.7B or Qwen3-4B for routing)
3. Add Text-to-Speech
4. Implement the full WAKE → LISTEN → UNDERSTAND → SEE → PLAN → ACT → VERIFY → COMPLETE loop

## Verification

To verify CPU usage is low:
1. Run the application
2. Open Task Manager
3. Look for Python process
4. CPU usage should be minimal (<5%) when idle