import argparse
import queue
import signal
import sys
import threading
import warnings

import numpy as np
import soundcard as sc
from soundcard import SoundcardRuntimeWarning
from transformers import pipeline
import whisper

from gui import create_subtitle_window, update_subtitle
from subtitles import recognize, translate



# 忽略 SoundcardRuntimeWarning 类型的警告
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)



################################################################################
# Arguments
################################################################################

SAMPLE_RATE = 16000
INTERVAL = 2
BUFFER_SIZE = 4096

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="medium")
args = parser.parse_args()


################################################################################
# Functions
################################################################################

def record():
    """
    This function is used to record audio and put it into the queue.
    """
    smoothing_filter = np.ones(100) / 100
    # start recording
    with sc.get_microphone(
        id=str(sc.default_speaker().name), include_loopback=True
    ).recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        buffer_end = 0
        while True:
            while buffer_end < SAMPLE_RATE * INTERVAL:
                data = mic.record(BUFFER_SIZE)
                audio[buffer_end : buffer_end + len(data)] = data.reshape(-1)
                buffer_end += len(data)

            # find silent periods
            silence_start = buffer_end * 4 // 5
            vol = np.convolve(audio[silence_start:buffer_end] ** 2, smoothing_filter, "same")
            silence_start += vol.argmin()
            audio_queue.put(audio[:silence_start])

            audio_prev = audio
            audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
            audio[: buffer_end - silence_start] = audio_prev[silence_start:buffer_end]
            buffer_end = buffer_end - silence_start


################################################################################
# Transcription
################################################################################

# load the model
print("Loading model...")
model = whisper.load_model(args.model)
translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-zh")
print("Done")

# initialize the queue and the filter
audio_queue = queue.Queue()


subtitle_en_queue = queue.Queue()
subtitle_zh_queue = queue.Queue()
tobetranslated_queue = queue.Queue()


# start recording, translating and recognizing audio in separate threads
th_recognize = threading.Thread(target=recognize, args=(model, audio_queue, subtitle_zh_queue, subtitle_en_queue, tobetranslated_queue), daemon=True)
th_recognize.start()
th_translate = threading.Thread(target=translate, args=(translator, tobetranslated_queue, subtitle_zh_queue), daemon=True)
th_translate.start()
th_record = threading.Thread(target=record, daemon=True)
th_record.start()


################################################################################
# GUI
################################################################################

window, text_en, text_zh, font_name = create_subtitle_window()


# 在主线程中启动字幕更新
update_subtitle(text_en, text_zh, subtitle_en_queue, subtitle_zh_queue)  




# Use Ctrl+C to quit the program
def signal_handler(sig, frame):
    window.quit()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

window.mainloop()
