"""
Generate positive and negative samples for hey beast wake word training
"""
import os
import uuid
import numpy as np
from scipy.io import wavfile
from piper.voice import PiperVoice
import scipy.signal
import tempfile

def generate_hey_beast_positive_samples(output_dir, num_samples=50):
    """Generate positive samples of 'hey beast' using Piper TTS"""
    print(f"Generating {num_samples} positive samples of 'hey beast'...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load Piper voice
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'piper_voices', 'en_US-lessac-medium.onnx')
    try:
        voice = PiperVoice.load(model_path)
    except Exception as e:
        print(f"Failed to load Piper voice: {e}")
        return []

    generated_files = []

    for i in range(num_samples):
        try:
            # Vary the speech parameters for diversity
            noise_scale = 0.3 + np.random.uniform(0, 0.4)  # 0.3-0.7
            noise_scale_w = 0.6 + np.random.uniform(0, 0.4)  # 0.6-1.0
            length_scale = 0.8 + np.random.uniform(0, 0.4)  # 0.8-1.2

            # Synthesize "hey beast"
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_w_scale=noise_scale_w
            )
            audio_generator = voice.synthesize("hey beast", syn_config=syn_config)
            # Collect all audio chunks
            audio_chunks = list(audio_generator)
            if audio_chunks:
                audio = np.concatenate([chunk.audio_int16_array for chunk in audio_chunks])
            else:
                # Fallback if no audio generated
                audio = np.array([], dtype=np.int16)

            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)

            # Resample to 16000 Hz if needed (Piper outputs 22050 Hz)
            if len(audio_int16) > 0:
                # Calculate resampling ratio
                orig_rate = 22050
                target_rate = 16000
                num_samples_target = int(len(audio_int16) * target_rate / orig_rate)

                # Resample using scipy
                if num_samples_target > 0:
                    audio_resampled = scipy.signal.resample(audio_int16, num_samples_target)
                    audio_int16 = np.clip(audio_resampled, -32768, 32767).astype(np.int16)

            # Save file
            filename = f"hey_beast_{i:03d}_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(output_dir, filename)
            wavfile.write(filepath, 16000, audio_int16)
            generated_files.append(filepath)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} positive samples")

        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

    print(f"Generated {len(generated_files)} positive samples")
    return generated_files

def generate_negative_samples(output_dir, num_samples=50):
    """Generate negative samples (random speech, noise, etc.)"""
    print(f"Generating {num_samples} negative samples...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load Piper voice for generating random speech
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'piper_voices', 'en_US-lessac-medium.onnx')
    try:
        voice = PiperVoice.load(model_path)
        voice_available = True
    except Exception as e:
        print(f"Failed to load Piper voice for negative samples: {e}")
        voice_available = False

    generated_files = []

    # Some random phrases for negative samples (not containing "hey beast")
    negative_phrases = [
        "hello world", "good morning", "how are you", "what time is it",
        "open the window", "play some music", "set a timer", "remind me",
        "what's the weather", "turn on the lights", "stop", "cancel",
        "never mind", "hey there", "hi buddy", "okay google", "alexa"
    ]

    for i in range(num_samples):
        try:
            if voice_available and i < num_samples // 2:
                # Generate random speech for first half
                phrase_idx = i % len(negative_phrases)
                phrase = negative_phrases[phrase_idx]

                # Vary speech parameters
                noise_scale = 0.1 + np.random.uniform(0, 0.5)  # 0.1-0.6
                noise_scale_w = 0.5 + np.random.uniform(0, 0.5)  # 0.5-1.0
                length_scale = 0.7 + np.random.uniform(0, 0.6)  # 0.7-1.3

                from piper.config import SynthesisConfig
                syn_config = SynthesisConfig(
                    noise_scale=noise_scale,
                    length_scale=length_scale,
                    noise_w_scale=noise_scale_w
                )
                audio_generator = voice.synthesize(phrase, syn_config=syn_config)
                # Collect all audio chunks
                audio_chunks = list(audio_generator)
                if audio_chunks:
                    audio = np.concatenate([chunk.audio_int16_array for chunk in audio_chunks])
                else:
                    # Fallback if no audio generated
                    audio = np.array([], dtype=np.int16)
            else:
                # Generate noise/background for second half
                # Create various types of noise
                duration = 1.0  # 1 second
                sample_rate = 16000
                samples = int(duration * sample_rate)

                # Choose noise type randomly
                noise_type = np.random.choice(['white', 'pink', 'babble', 'street'])

                if noise_type == 'white':
                    audio = np.random.normal(0, 0.1, samples)
                elif noise_type == 'pink':
                    # Approximate pink noise
                    white_noise = np.random.normal(0, 0.1, samples)
                    # Simple pink noise filter (approximation)
                    b, a = scipy.signal.butter(1, 0.05, btype='low')
                    audio = scipy.signal.filtfilt(b, a, white_noise)
                elif noise_type == 'babble':
                    # Multi-speaker babble approximation
                    t = np.linspace(0, duration, samples, False)
                    audio = np.zeros(samples)
                    for _ in range(3):  # 3 overlapping speakers
                        freq1 = 80 + np.random.uniform(0, 40)
                        freq2 = 120 + np.random.uniform(0, 60)
                        audio += 0.1 * np.sin(2*np.pi*freq1*t) * np.exp(-t*0.5 + np.random.uniform(0, 2))
                        audio += 0.08 * np.sin(2*np.pi*freq2*t) * np.exp(-t*0.3 + np.random.uniform(0, 2))
                    audio += np.random.normal(0, 0.05, samples)  # Add some noise
                else:  # street
                    # Street noise approximation
                    audio = np.random.normal(0, 0.08, samples)
                    # Add occasional transient sounds
                    for _ in range(np.random.randint(2, 5)):
                        pos = np.random.randint(0, samples - 1000)
                        length = np.random.randint(100, 800)
                        transient = np.random.normal(0, 0.3, length) * np.exp(-np.linspace(0, 3, length//2))
                        audio[pos:pos+length//2] += transient[:length//2]

            # Convert to 16-bit PCM
            audio_int16 = np.clip(audio, -1, 1) * 32767
            audio_int16 = audio_int16.astype(np.int16)

            # Save file
            filename = f"negative_{i:03d}_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(output_dir, filename)
            wavfile.write(filepath, 16000, audio_int16)
            generated_files.append(filepath)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} negative samples")

        except Exception as e:
            print(f"Error generating negative sample {i}: {e}")
            continue

    print(f"Generated {len(generated_files)} negative samples")
    return generated_files

if __name__ == "__main__":
    # Generate positive samples
    positive_dir = os.path.join(os.path.dirname(__file__), "positive_samples")
    positive_files = generate_hey_beast_positive_samples(positive_dir, num_samples=30)

    # Generate negative samples
    negative_dir = os.path.join(os.path.dirname(__file__), "negative_samples")
    negative_files = generate_negative_samples(negative_dir, num_samples=30)

    print(f"\nSummary:")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print(f"Positive samples directory: {positive_dir}")
    print(f"Negative samples directory: {negative_dir}")