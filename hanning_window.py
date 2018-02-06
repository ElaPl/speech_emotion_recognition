import math


class HanningWindow:
    """
    Klasa reprezentuje funckję Hanninga o konkrentej wartości
    """
    def __init__(self, size):
        """
        Funkcja tworzy okno Hanninga o podanym rozmiarze

        :param size: rozmiar okna
        """
        self.hanning_window = []
        for i in range(0, size):
            self.hanning_window.append(0.5*(math.cos((2 * math.pi * i)/(size-1))))

    def plot(self, signal):
        """
        Funkcja przemnaża podany jako argument sygnał przez okno Hanninga

        :param signal: Fragment sygnału dźwiękowego

        :return: sygnał będący wynikiem przemnożenia podanego sygnału przez okno Hanninga
        """
        windowing_signal = []
        for i in range(0, len(signal)):
            windowing_signal.append(signal[i] * self.hanning_window[i])

        return windowing_signal
