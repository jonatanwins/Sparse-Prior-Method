from dataclasses import dataclass
import numpy as np


@dataclass
class SoundSource:
    distance: float
    angle: float
    frequency: float
    amplitude: float = 1.0
    phase: float = 0.0
    function: callable = np.sin

    def get_waveform(self, t):
        return self.amplitude * self.function(
            2 * np.pi * self.frequency * t + self.phase
        )

    def get_position(self):
        x = self.distance * np.cos(self.angle)
        y = self.distance * np.sin(self.angle)
        return x, y
