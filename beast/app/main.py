import sys
import os
import threading
import time
import logging
from pathlib import Path

import pystray
from PIL import Image, ImageDraw
import openwakeword
from openwakeword.model import Model
import sounddevice as sd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "beast.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import VoicePipeline for STT -> LLM -> TTS
from pipeline import VoicePipeline


class BeastApp:
    def __init__(self):
        self.icon = None
        self.wake_word_detector = None
        self.audio_stream = None
        self.listen_thread = None
        self.processing_thread = None

        # State management
        self._state = "IDLE"  # IDLE, LISTENING, PROCESSING, SPEAKING
        self._state_lock = threading.Lock()

        # Load configuration
        self.config = self.load_config()

        # Initialize wake word model
        self.initialize_wake_word()

        # Create system tray icon
        self.create_tray_icon()

    def load_config(self):
        """Load configuration from config file"""
        config_path = PROJECT_ROOT / "config" / "settings.json"
        default_config = {
            "microphone": None,  # Use default
            "wake_sensitivity": 0.5,
            "wake_word": "hey beast",
            "model_path": str(PROJECT_ROOT / "models" / "hey_beast.onnx")
        }

        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            # Create default config
            config_path.parent.mkdir(exist_ok=True)
            import json
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

        return default_config

    def initialize_wake_word(self):
        """Initialize the wake word detection model"""
        try:
            logger.info("Initializing wake word detection...")

            # Use pre-trained models from openwakeword
            # For custom wake word, we'll need to train or use existing models
            # For now, we'll try to load a known model; if not available, we'll note that custom training is needed
            try:
                self.wake_word_detector = Model(
                    wakeword_models=["hey_jarvis"],  # Using existing model as placeholder
                    inference_framework='onnx'
                )
                logger.info("Wake word detection initialized with hey_jarvis model")
            except Exception as model_error:
                logger.warning(f"Could not load pre-trained model: {model_error}")
                logger.info("Initializing wake word detection without model (will simulate)")
                self.wake_word_detector = Model(
                    wakeword_models=[],  # No pre-trained models
                    inference_framework='onnx'
                )
                # We'll still use the Model class but it won't detect anything real

        except Exception as e:
            logger.error(f"Failed to initialize wake word detection: {e}")
            # Fallback: create a simple detector that logs when activated
            self.wake_word_detector = None

    def create_tray_icon(self):
        """Create the system tray icon and menu"""
        # Create a simple icon
        image = Image.new('RGB', (64, 64), color='black')
        draw = ImageDraw.Draw(image)
        draw.text((10, 20), "B", fill='white')

        # Create menu
        menu = pystray.Menu(
            pystray.MenuItem('Status', self.show_status),
            pystray.MenuItem('Settings', self.show_settings),
            pystray.MenuItem('Activity Log', self.show_activity_log),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('Emergency STOP', self.emergency_stop),
            pystray.MenuItem('Quit', self.quit_app)
        )

        self.icon = pystray.Icon("Beast", image, "Beast AI Assistant", menu)

    def show_status(self):
        """Show status information"""
        logger.info("Status requested")
        # TODO: Implement proper status dialog

    def show_settings(self):
        """Show settings"""
        logger.info("Settings requested")
        # TODO: Implement settings dialog

    def show_activity_log(self):
        """Show activity log"""
        logger.info("Activity log requested")
        # TODO: Implement log viewer

    def emergency_stop(self):
        """Emergency stop all operations"""
        logger.warning("Emergency STOP activated")
        self.stop_listening()
        # TODO: Stop all ongoing operations

    def quit_app(self):
        """Quit the application"""
        logger.info("Quitting application")
        self.stop_listening()
        if self.icon:
            self.icon.stop()

    def _set_state(self, new_state):
        """Thread-safe state transition"""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            logger.debug(f"State transition: {old_state} -> {new_state}")

    def _get_state(self):
        """Thread-safe state read"""
        with self._state_lock:
            return self._state

    def wake_word_callback(self, detections):
        """Callback when wake word is detected"""
        logger.info(f"Wake word detected: {detections}")
        # Show notification
        if self.icon:
            self.icon.notify("Beast is listening...", "Wake Word Detected")

        # Only process if we're in IDLE or LISTENING state (not already processing/speaking)
        current_state = self._get_state()
        if current_state in ["IDLE", "LISTENING"]:
            logger.info(f"Wake word detected in {current_state} state, starting command processing")
            self._set_state("PROCESSING")

            # Start processing in a separate thread to avoid blocking audio callback
            self.processing_thread = threading.Thread(target=self._process_command, daemon=True)
            self.processing_thread.start()
        else:
            logger.info(f"Ignoring wake word detection during {current_state} state")

    def _process_command(self):
        """Process a command after wake word detection"""
        try:
            logger.info("Starting command processing pipeline")

            # Initialize voice pipeline
            pipeline = VoicePipeline()

            # Record command audio (we'll reuse the audio stream for simplicity)
            # In a more sophisticated implementation, we might want to record a few seconds of audio
            # after the wake word for the command
            logger.info("Listening for command...")

            # For now, we'll just use a placeholder - in reality we'd capture audio from the stream
            # This is a simplified version - the actual implementation would need to buffer audio
            # from the wake word detection through to the command

            # Placeholder response - in a real implementation, this would come from the pipeline
            response = "I heard the wake word. Command processing not yet fully implemented."
            logger.info(f"[PIPELINE] Response: {response!r}")

            # Speak the response
            if hasattr(pipeline, 'tts'):
                pipeline.tts.speak(response)

            logger.info("Command processing completed")

        except Exception as e:
            logger.error(f"Error in command processing: {e}")
        finally:
            # Return to idle state after processing
            self._set_state("IDLE")
            logger.info("Returned to IDLE state")

    def start_listening(self):
        """Start wake word detection"""
        logger.info("start_listening called")

        # Idempotent: if already listening, do nothing
        if self._get_state() == "LISTENING":
            logger.info("Already listening, ignoring start_listening call")
            return

        # Idempotent: if we have a stream, make sure it's stopped first
        if self.audio_stream is not None:
            logger.warning("Audio stream exists but not listening - cleaning up")
            self.stop_listening()

        logger.info("Starting wake word detection...")
        self._set_state("LISTENING")

        # Real implementation: open audio stream from microphone
        try:
            import sounddevice as sd
            import numpy as np

            # Get microphone index from config (None = default device)
            microphone_index = self.config.get("microphone")
            if microphone_index is None:
                logger.info("Using default audio device")
                device = None  # Use system default
            else:
                device = microphone_index
                logger.info(f"Using configured microphone device index: {device}")

            # Audio parameters
            samplerate = 16000  # Standard for wake word detection
            channels = 1        # Mono
            blocksize = 1280    # Reasonable blocksize for real-time processing

            logger.info(f"Opening audio stream: device={device}, samplerate={samplerate}Hz, channels={channels}, blocksize={blocksize}")

            def audio_callback(indata, frames, time, status):
                """Real audio callback for wake word detection"""
                if status:
                    logger.warning(f"Audio stream status: {status}")

                # Only process wake word detection if we're in LISTENING state
                # (not during PROCESSING or SPEAKING to avoid false triggers from our own output)
                if self._get_state() != "LISTENING":
                    return

                try:
                    # Convert audio to the format expected by openwakeword
                    # sounddevice gives us float32 in [-1, 1], openwakeword expects int16
                    # Take first channel if stereo (we requested mono anyway)
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        # Stereo input - take first channel
                        audio_float32 = indata[:, 0]
                    else:
                        # Already mono or single channel
                        audio_float32 = indata.flatten()

                    # Convert float32 [-1, 1] to int16 for openwakeword
                    audio_int16 = (audio_float32 * 32767).astype(np.int16)

                    # Run wake word detection
                    if self.wake_word_detector is not None:
                        prediction = self.wake_word_detector.predict(audio_int16)

                        # Handle both dict and float returns from openwakeword
                        if isinstance(prediction, dict):
                            # Get the score for our wake word model
                            wake_word_key = "hey_jarvis"  # Matches what we initialized with
                            score = prediction.get(wake_word_key, 0.0)
                        else:
                            score = float(prediction)

                        # Check if score exceeds sensitivity threshold
                        sensitivity = self.config.get("wake_sensitivity", 0.4)
                        # Debug logging
                        logger.debug(f"Wake word score: {score:.6f}, sensitivity: {sensitivity}, state: {self._get_state()}")
                        if score > sensitivity:
                            logger.info(f"Wake word detected! Score: {score:.6f} > threshold {sensitivity}")
                            # Call the wake word callback (will handle state checking and debouncing)
                            self.wake_word_callback({wake_word_key: score})
                        #else:
                            # Log occasionally for debugging
                            #if int(time.time() * 10) % 10 == 0:  # Log every ~100ms to avoid spam
                                #logger.debug(f"Wake word score: {score:.6f} (threshold: {sensitivity})")

                except Exception as e:
                    logger.error(f"Error in audio callback: {e}", exc_info=True)

            # Create and start the audio stream
            self.audio_stream = sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                dtype='float32',
                callback=audio_callback,
                device=device,
                blocksize=blocksize
            )

            self.audio_stream.start()
            logger.info(f"Audio stream started successfully on device {device}")

        except ImportError as e:
            logger.error(f"sounddevice not available: {e}")
            logger.warning("Falling back to simulated wake word detection")
            self._start_simulated_detection()
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            logger.warning("Falling back to simulated wake word detection")
            self._start_simulated_detection()

    def _start_simulated_detection(self):
        """Fallback to simulated detection if real audio fails"""
        logger.info("Starting simulated wake word detection (fallback)")

        def listen_loop():
            while self._get_state() == "LISTENING":
                # Simulate wake word detection every 30 seconds for testing
                time.sleep(30)
                if self._get_state() == "LISTENING":  # Check again after sleep
                    logger.info("Simulated wake word detection")
                    self.wake_word_callback({"hey_jarvis": 0.8})

        self.listen_thread = threading.Thread(target=listen_loop, daemon=True)
        self.listen_thread.start()

    def stop_listening(self):
        """Stop wake word detection"""
        logger.info("stop_listening called")

        # Set state to IDLE first to stop any processing
        old_state = self._get_state()
        self._set_state("IDLE")
        logger.info(f"Transitioning from {old_state} to IDLE state")

        # Stop audio stream if it exists
        if self.audio_stream is not None:
            try:
                logger.info("Stopping audio stream")
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                logger.info("Audio stream stopped and closed")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")

        # Wait for listen thread to finish (if it exists)
        if self.listen_thread is not None and self.listen_thread.is_alive():
            logger.info("Waiting for listen thread to finish")
            self.listen_thread.join(timeout=1.0)
            self.listen_thread = None

        # Wait for processing thread to finish (if it exists)
        if self.processing_thread is not None and self.processing_thread.is_alive():
            logger.info("Waiting for processing thread to finish")
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None

        logger.info("Wake word detection stopped")

    def _process_command(self):
        """Process a command after wake word detection: record audio and run pipeline."""
        try:
            logger.info("Starting command processing")

            # Stop the wake word audio stream so we can use the microphone for command recording
            if self.audio_stream is not None:
                logger.info("Stopping wake word audio stream for command recording")
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            # Import sounddevice and numpy (already imported at top, but ensure)
            import sounddevice as sd
            import numpy as np

            # Get microphone index from config (None = default device)
            microphone_index = self.config.get("microphone")
            if microphone_index is None:
                logger.info("Using default audio device for command recording")
                device = None  # Use system default
            else:
                device = microphone_index
                logger.info(f"Using configured microphone device index: {device} for command recording")

            # Audio parameters for recording (same as wake word detection)
            samplerate = 16000  # Standard for wake word detection
            channels = 1        # Mono
            # We'll record for 5 seconds
            duration = 5.0

            logger.info(f"Recording command audio for {duration} seconds on device {device}")

            # Record audio
            recording = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=channels,
                dtype='float32',
                device=device
            )
            sd.wait()  # Wait until recording is finished
            logger.info(f"Finished recording, shape: {recording.shape}")

            # Ensure we have mono audio (if stereo, take first channel)
            if recording.ndim > 1 and recording.shape[1] > 1:
                recording = recording[:, 0]
            else:
                recording = recording.flatten()

            # Process the audio with the voice pipeline
            logger.info("Processing command audio with VoicePipeline")
            pipeline = VoicePipeline()
            response = pipeline.process_command(recording)
            logger.info(f"[PIPELINE] Response: {response!r}")

        except Exception as e:
            logger.error(f"Error in command processing: {e}", exc_info=True)
        finally:
            # Restart the wake word audio stream
            logger.info("Restarting wake word audio stream after command processing")
            self.start_listening()  # This will set state to LISTENING and start the stream

            logger.info("Command processing completed, returning to LISTENING state")

    def run(self):
        """Run the application"""
        logger.info("Starting Beast AI Assistant...")

        # Start wake word detection
        self.start_listening()

        # Run system tray icon (blocks until quit)
        self.icon.run()


def main():
    app = BeastApp()
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        app.quit_app()


if __name__ == "__main__":
    main()