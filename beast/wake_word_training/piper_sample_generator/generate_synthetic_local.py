import os
import sys
# Ensure we are in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Add the parent directories to sys.path if needed, but we can import directly
sys.path.insert(0, script_dir)
from generate_samples import generate_samples

text = ['Hey Beast']
max_samples = 250
batch_size = 10
noise_scales = [0.98]
noise_scale_ws = [0.98]
length_scales = [0.75, 1.0, 1.25]
output_dir = os.path.join(script_dir, '..', '..', 'training_data', 'positive')
file_names = [f'hey_beast_synth_{i:03d}.wav' for i in range(1, max_samples + 1)]

print(f'Generating {max_samples} synthetic samples...')
print(f'Output dir: {output_dir}')
generated = generate_samples(
    text=text,
    max_samples=max_samples,
    batch_size=batch_size,
    noise_scales=noise_scales,
    noise_scale_ws=noise_scale_ws,
    length_scales=length_scales,
    output_dir=output_dir,
    auto_reduce_batch_size=True,
    file_names=file_names
)
print(f'Generated {len(generated)} files.')