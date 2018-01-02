import struct


class WavIterator:
    def __init__(self, wav_file, frame_count):
        self.wav_file = wav_file
        self.frame_count = frame_count
        self.sizes = {1: 'B', 2: 'h', 4: 'i'}
        self.fmt_size = self.sizes[self.wav_file.getsampwidth()]
        self.fmt = "<" + self.fmt_size * self.frame_count * self.wav_file.getnchannels()

    def __iter__(self):
        return self

    def __next__(self):
        if self.wav_file.tell() >= self.wav_file.getnframes():
            raise StopIteration()
        else:
            if self.wav_file.getnframes() - self.wav_file.tell() >= self.frame_count:
                decoded = struct.unpack(self.fmt, self.wav_file.readframes(self.frame_count))
                decoded_on_channel = []
                for i in range(0, len(decoded), self.wav_file.getnchannels()):
                    decoded_on_channel.append(decoded[i])
                return decoded_on_channel
            else:
                tmp_size = self.wav_file.getnframes() - self.wav_file.tell()
                tmp_fmt = "<" + self.fmt_size * tmp_size * self.wav_file.getnchannels()
                decoded = struct.unpack(tmp_fmt, self.wav_file.readframes(tmp_size))
                decoded_on_channel = []
                for i in range(0, len(decoded), self.wav_file.getnchannels()):
                    decoded_on_channel.append(decoded[i])
                return decoded_on_channel
