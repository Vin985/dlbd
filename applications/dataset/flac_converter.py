import soundfile as sf
import pyflac


class FlacConverter:
    def __init__(self):
        self.idx = 0
        self.output_file = None
        self.total_bytes = 0

    def encode(self, file_path, file_dest, block_size=0, compression_level=5):
        print(f"Compressing {file_path} into {file_dest}")
        self.total_bytes = 0
        self.output_file = open(file_dest, "wb")
        data, sr = sf.read(str(file_path), dtype="int16", always_2d=True)
        encoder = pyflac.StreamEncoder(
            write_callback=self.encoder_callback,
            sample_rate=sr,
            blocksize=block_size,
            compression_level=compression_level,
        )
        encoder.process(data)
        encoder.finish()
        print(
            f"Compression successful with ratio = {self.total_bytes / data.nbytes * 100:.2f}%"
        )

    def encoder_callback(self, buffer, num_bytes, num_samples, current_frame):
        self.total_bytes += num_bytes
        if self.output_file:
            self.output_file.write(buffer)
            self.output_file.flush()


def main():
    encoder = FlacConverter()
    file_path = "/mnt/win/UMoncton/Doctorat/dev/pysoundplayer/examples/example.wav"
    file_dest = "./compressed.flac"
    encoder.encode(file_path, file_dest)


if __name__ == "__main__":
    main()
