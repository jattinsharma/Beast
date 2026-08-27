import inspect
from piper.config import SynthesisConfig

# Get the signature of SynthesisConfig.__init__
sig = inspect.signature(SynthesisConfig.__init__)
print("SynthesisConfig.__init__ signature:")
print(sig)

# Get the parameters
params = sig.parameters
print("\nParameters:")
for name, param in params.items():
    if name != 'self':
        print(f"  {name}: {param}")