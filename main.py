from faster_whisper import WhisperModel
import numpy as np
import pyaudio
import time

CHUNK_SIZE = 16000 * 5
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 0.01  # Adjust this threshold based on your needs
SILENCE_DURATION = 2  # Duration to consider as silence (in seconds)

model_size = "tiny.en"

def send_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    try:
        while True:
            data = stream.read(CHUNK_SIZE)
            numpy_data = np.frombuffer(data, dtype=np.int16)  # Read as int16
            numpy_data = numpy_data.astype(np.float32) / 32768.0  # Convert to float32 and normalize

            segments, info = model.transcribe(numpy_data, beam_size=5, language="en")

            has_speech = False
            for segment in segments:
                if segment.text.strip():
                    print(segment.text)
                    has_speech = True

            if not has_speech:
                silence_start_time = time.time()
                while not has_speech:
                    data = stream.read(CHUNK_SIZE)
                    numpy_data = np.frombuffer(data, dtype=np.int16)  # Read as int16
                    numpy_data = numpy_data.astype(np.float32) / 32768.0  # Convert to float32 and normalize

                    segments, info = model.transcribe(numpy_data, beam_size=5, language="en")
                    for segment in segments:
                        if segment.text.strip():
                            print(segment.text)
                            has_speech = True
                            break


                        # print("Holding until next voice input...")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

send_audio()
