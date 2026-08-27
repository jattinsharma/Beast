"""
Create false positive validation dataset for openWakeWord training
"""
import os
import numpy as np
from scipy.io import wavfile
import uuid
import sys

# Add the beast directory to path so we can import openwakeword utilities
sys.path.append('C:/Beast/beast')
sys.path.append('C:/Beast/beast/venv/Lib/site-packages')

from openwakeword.utils import AudioFeatures


def generate_false_positive_validation(output_path, duration_seconds=60*60*11.3):  # ~11.3 hours as used in openWakeWord
    """
    Generate false positive validation data - background noise and random speech
    Then compute embeddings using the same pipeline as training data
    """
    print(f"Generating false positive validation data ({duration_seconds/3600:.1f} hours)...")

    sample_rate = 16000
    total_samples = int(duration_seconds * sample_rate)

    # We'll generate in chunks to avoid memory issues
    chunk_duration = 10  # 10 seconds per chunk
    chunk_samples = int(chunk_duration * sample_rate)
    num_chunks = int(np.ceil(total_samples / chunk_samples))

    all_chunks = []

    for i in range(num_chunks):
        print(f"Generating chunk {i+1}/{num_chunks}")

        # Generate background noise
        noise = np.random.normal(0, 0.05, chunk_samples)

        # Add occasional speech-like bursts
        for _ in range(np.random.randint(0, 3)):  # 0-2 speech bursts per chunk
            burst_start = np.random.randint(0, chunk_samples - int(0.5*sample_rate))
            burst_length = np.random.randint(int(0.2*sample_rate), int(0.8*sample_rate))

            t = np.linspace(0, burst_length/sample_rate, burst_length, False)
            # Create speech-like sound with formants
            burst = (
                0.3 * np.sin(2*np.pi*120*t) * np.exp(-t*2) +
                0.2 * np.sin(2*np.pi*250*t) * np.exp(-t*1.5) +
                0.15 * np.sin(2*np.pi*400*t) * np.exp(-t*1) +
                0.1 * np.sin(2*np.pi*600*t) * np.exp(-t*0.8)
            )
            noise[burst_start:burst_start+burst_length] += burst * 0.3

        # Add some environmental sounds
        for _ in range(np.random.randint(0, 2)):
            event_start = np.random.randint(0, chunk_samples - int(0.3*sample_rate))
            event_length = np.random.randint(int(0.1*sample_rate), int(0.4*sample_rate))

            # Click-like or pop-like sounds
            event = np.random.normal(0, 0.5, event_length) * np.exp(-np.linspace(0, 5, event_length))
            noise[event_start:event_start+event_length] += event * 0.2

        # Normalize to prevent clipping
        if np.max(np.abs(noise)) > 0:
            noise = noise / np.max(np.abs(noise)) * 0.8

        all_chunks.append(noise.astype(np.float32))

    # Concatenate all chunks
    validation_audio = np.concatenate(all_chunks)[:total_samples]
    print(f"Generated raw audio: {validation_audio.shape}, Duration: {len(validation_audio)/sample_rate:.1f} seconds")

    # Now compute embeddings using the same pipeline as training data
    print("Computing embeddings...")

    # Initialize AudioFeatures (same parameters as used in training)
    af = AudioFeatures(device='cpu', ncpu=1)

    # Process audio in chunks suitable for embedding computation
    # Use 2-second chunks to match the expected feature shape
    embed_chunk_duration = 2.0  # seconds
    embed_chunk_samples = int(embed_chunk_duration * sample_rate)
    print(f"Embedding chunk size: {embed_chunk_samples} samples ({embed_chunk_duration}s)")

    # Get expected embedding shape for this duration
    embed_shape = af.get_embedding_shape(embed_chunk_duration)
    print(f"Expected embedding shape per chunk: {embed_shape}")  # Should be (16, 96) for 2 seconds

    # Convert float audio to int16 PCM as required by AudioFeatures
    validation_audio_int16 = (validation_audio * 32767).astype(np.int16)

    # Reshape into chunks
    num_embed_chunks = len(validation_audio_int16) // embed_chunk_samples
    validation_audio_int16 = validation_audio_int16[:num_embed_chunks * embed_chunk_samples]
    audio_chunks = validation_audio_int16.reshape((num_embed_chunks, embed_chunk_samples))
    print(f"Audio chunks shape: {audio_chunks.shape}")

    # Compute embeddings
    print("Running embedding computation...")
    embeddings = af.embed_clips(audio_chunks, batch_size=4)  # Smaller batch size for memory
    print(f"Computed embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved false positive validation embeddings to {output_path}")
    print(f"Shape: {embeddings.shape}, Duration equivalent: {embeddings.shape[0] * embed_chunk_duration:.1f} seconds")


if __name__ == "__main__":
    output_path = "C:\\Beast\\beast\\wake_word_training\\false_positive_validation.npy"
    # Generate 10 minutes worth for testing (same as original script)
    generate_false_positive_validation(output_path, duration_seconds=60*10)  # 10 minutes for testing