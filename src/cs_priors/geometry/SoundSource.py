from dataclasses import dataclass
import numpy as np


# @dataclass
# class SoundSource:
#     distance: float
#     angle: float
#     frequency: float | list[float]  # Single or multiple frequencies
#     amplitude: float | list[float] = 1.0
#     phase: float | list[float] = 0.0
#     function = np.sin

#     # def get_waveform(self, t):
#     #     if isinstance(self.frequency, (list, np.ndarray)):
#     #         # Sum multiple frequency components
#     #         freqs = np.array(self.frequency)
#     #         amps = np.atleast_1d(self.amplitude)
#     #         phases = np.atleast_1d(self.phase)
#     #         return sum(
#     #             a * self.function(2 * np.pi * f * t + p)
#     #             for f, a, p in zip(freqs, amps, phases)
#     #         )
#     #     else:
#     #         return self.amplitude * self.function(
#     #             2 * np.pi * self.frequency * t + self.phase
#     #         )

#     def get_position(self):
#         x = self.distance * np.cos(self.angle)
#         y = self.distance * np.sin(self.angle)
#         return x, y


@dataclass
class SoundSource:
    distance: float
    angle: float
    time_series: np.ndarray | None

    def get_position(self):
        x = self.distance * np.cos(self.angle)
        y = self.distance * np.sin(self.angle)
        return x, y
