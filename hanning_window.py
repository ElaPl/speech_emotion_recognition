import math


class HanningWindow:
    def __init__(self, size):
        self.hanning_window = []
        for i in range(0, size):
            self.hanning_window.append(0.5*(math.cos((2 * math.pi * i)/(size-1))))

    def plot(self, signal):
        windowing_signal = []
        for i in range(0, len(signal)):
            windowing_signal.append(signal[i] * self.hanning_window[i])

        return windowing_signal
