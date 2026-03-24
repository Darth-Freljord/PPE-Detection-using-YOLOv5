import numpy as np
import wave
import struct

freq = 440  # Hz (A4 note)
duration = 0.5  # seconds
sample_rate = 44100
amplitude = 32767  # Max for 16-bit PCM audio

# Generate time values
t = np.linspace(0, duration, int(sample_rate * duration), False)

# Generate sine wave
audio = amplitude * np.sin(2 * np.pi * freq * t)

# Convert float samples to 16-bit PCM format
audio_data = b''.join(struct.pack('<h', int(sample)) for sample in audio)

# Write to WAV file
with wave.open('beep.wav', 'w') as f:
    f.setnchannels(1)  # Mono
    f.setsampwidth(2)  # 16-bit audio
    f.setframerate(sample_rate)
    f.writeframes(audio_data)  # Write all at once

print("beep.wav successfully generated!")
