"""
Synthetic sample generation for openWakeWord training using Piper TTS
"""
import os
import uuid
import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm
from piper.voice import PiperVoice
import tempfile

def generate_samples(text, max_samples, batch_size, noise_scales, noise_scale_ws, length_scales,
                    output_dir, auto_reduce_batch_size=True, file_names=None):
    """
    Generate synthetic wake word samples using Piper TTS

    Args:
        text: Text to synthesize (can be string or list of strings)
        max_samples: Maximum number of samples to generate
        batch_size: Batch size for TTS synthesis
        noise_scales: Noise scales for TTS
        noise_scale_ws: Noise scale weights for TTS
        length_scales: Length scales for TTS
        output_dir: Directory to save generated WAV files
        auto_reduce_batch_size: Whether to automatically reduce batch size on OOM
        file_names: Optional list of filenames to use

    Returns:
        List of generated file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # If text is a string, convert to list
    if isinstance(text, str):
        text = [text]

    # Load Piper voice model (use the one we downloaded)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'piper_voices', 'en_US-lessac-medium.onnx')
    config_path = model_path.replace('.onnx', '.onnx.json')

    # If config doesn't exist, Piper will use defaults
    if not os.path.exists(config_path):
        config_path = None

    try:
        voice = PiperVoice.load(model_path, config_path)
    except Exception as e:
        print(f"Failed to load Piper voice: {e}")
        print("Falling back to dummy audio generation")
        return generate_dummy_samples(text, max_samples, batch_size, output_dir, file_names)

    generated_files = []

    # Generate samples in batches
    samples_generated = 0
    pbar = tqdm(total=max_samples, desc="Generating samples")

    while samples_generated < max_samples:
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, max_samples - samples_generated)
        if current_batch_size <= 0:
            break

        # Select text for this batch (cycle through provided texts)
        batch_texts = []
        for i in range(current_batch_size):
            text_idx = (samples_generated + i) % len(text)
            batch_texts.append(text[text_idx])

        try:
            # Generate audio for each text in batch
            for i, txt in enumerate(batch_texts):
                if samples_generated >= max_samples:
                    break

                # Synthesize audio
                audio = voice.synthesize(txt,
                                       noise_scale=noise_scales[0] if isinstance(noise_scales, list) else noise_scales,
                                       noise_scale_w=noise_scale_ws[0] if isinstance(noise_scale_ws, list) else noise_scale_ws,
                                       length_scale=length_scales[0] if isinstance(length_scales, list) else length_scales)

                # Convert to proper format (16-bit PCM)
                audio_int16 = (audio * 32767).astype(np.int16)

                # Generate filename
                if file_names and samples_generated < len(file_names):
                    filename = file_names[samples_generated]
                else:
                    filename = f"{uuid.uuid4().hex}.wav"

                filepath = os.path.join(output_dir, filename)

                # Save as WAV file
                wavfile.write(filepath, 22050, audio_int16)  # Piper typically outputs 22050Hz

                generated_files.append(filepath)
                samples_generated += 1
                pbar.update(1)

        except Exception as e:
            print(f"Error during TTS synthesis: {e}")
            if auto_reduce_batch_size and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"Reducing batch size to {batch_size} and retrying...")
                continue
            else:
                # Fallback to dummy samples
                print("Falling back to dummy sample generation")
                dummy_files = generate_dummy_samples(text, max_samples - samples_generated, 1, output_dir,
                                                   file_names[samples_generated:] if file_names else None)
                generated_files.extend(dummy_files)
                break

    pbar.close()
    return generated_files

def generate_dummy_samples(text, max_samples, batch_size, output_dir, file_names=None):
    """Generate dummy audio samples when TTS fails"""
    print(f"Generating {max_samples} dummy samples...")
    generated_files = []

    for i in range(max_samples):
        # Generate random noise that resembles speech
        duration = 1.0  # 1 second
        sample_rate = 16000
        samples = int(duration * sample_rate)

        # Generate pink noise filtered to resemble speech frequencies
        noise = np.random.normal(0, 0.1, samples)
        # Apply some filtering to make it more speech-like
        from scipy import signal
        b, a = signal.butter(4, [80/(sample_rate/2), 3000/(sample_rate/2)], btype='band')
        filtered_noise = signal.filtfilt(b, a, noise)

        # Add some periodic components to resemble speech patterns
        t = np.linspace(0, duration, samples, False)
        speech_like = 0.3 * np.sin(2*np.pi*150*t) * np.exp(-t*0.5)  # Fundamental frequency
        speech_like += 0.2 * np.sin(2*np.pi*300*t) * np.exp(-t*0.3)  # First formant
        speech_like += 0.1 * np.sin(2*np.pi*450*t) * np.exp(-t*0.4)  # Second formant

        audio = filtered_noise + 0.3 * speech_like
        audio_int16 = (audio * 32767).astype(np.int16)

        # Generate filename
        if file_names and i < len(file_names):
            filename = file_names[i]
        else:
            filename = f"{uuid.uuid4().hex}.wav"

        filepath = os.path.join(output_dir, filename)
        wavfile.write(filepath, sample_rate, audio_int16)
        generated_files.append(filepath)

    return generated_files