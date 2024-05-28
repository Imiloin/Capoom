import signal
import sys
import whisper
import soundcard as sc
import threading
import queue
import numpy as np
import argparse
import warnings
from soundcard import SoundcardRuntimeWarning
import tkinter as tk



# 忽略 SoundcardRuntimeWarning 类型的警告
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)



################################################################################
# Arguments
################################################################################

SAMPLE_RATE = 16000
INTERVAL = 5
BUFFER_SIZE = 4096

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="medium")
args = parser.parse_args()


################################################################################
# Transcription
################################################################################

# options = whisper.DecodingOptions(task="translate")
def recognize():
    """
    This function is used to recognize audio and print the recognized text.
    """
    result = None
    options = whisper.DecodingOptions()
    while True:
        audio = audio_data.get()
        if (audio**2).max() > 0.001:
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)
            lang = max(probs, key=probs.get)
            if (not result or prev_lang != "zh") and lang == "zh":
                options = whisper.DecodingOptions(prompt="我在说简体中文哦，")
            prev_lang = lang

            # options = whisper.DecodingOptions(task="transcribe", language='zh') # this kinda sucks

            # decode the audio
            result = whisper.decode(model, mel, options)

            options = whisper.DecodingOptions(prompt=result.text)

            # print the recognized text
            print(f"{lang}: {result.text}")
            subtitle_queue.put(result.text)


def record():
    """
    This function is used to record audio and put it into the queue.
    """
    # start recording
    with sc.get_microphone(
        id=str(sc.default_speaker().name), include_loopback=True
    ).recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        n = 0
        while True:
            while n < SAMPLE_RATE * INTERVAL:
                data = mic.record(BUFFER_SIZE)
                audio[n : n + len(data)] = data.reshape(-1)
                n += len(data)

            # find silent periods
            m = n * 4 // 5
            vol = np.convolve(audio[m:n] ** 2, b, "same")
            m += vol.argmin()
            audio_data.put(audio[:m])

            audio_prev = audio
            audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
            audio[: n - m] = audio_prev[m:n]
            n = n - m


# load the model
print("Loading model...")
model = whisper.load_model(args.model)
print("Done")

# initialize the queue and the filter
audio_data = queue.Queue()
b = np.ones(100) / 100


# start recording and recognizing audio in separate threads
th_recognize = threading.Thread(target=recognize, daemon=True)
th_recognize.start()
th_record = threading.Thread(target=record, daemon=True)
th_record.start()


################################################################################
# GUI
################################################################################


def create_subtitle_window():
    window = tk.Tk()
    window.overrideredirect(True)  # 移除窗口边框
    window.attributes("-alpha", 0.7)  # 设置窗口透明度
    window.geometry("500x100+300+300")  # 设置窗口大小和位置
    window.configure(background="black")  # 设置窗口背景色

    label = tk.Label(window, text="", fg="white", bg="black", font=("Helvetica", 16))
    label.pack()

    return window, label


def update_subtitle(label, subtitle_queue):
    try:
        text = subtitle_queue.get_nowait()
        label.config(text=text)
    except queue.Empty:
        pass
    label.after(100, update_subtitle, label, subtitle_queue)  # 每100毫秒更新一次字幕


window, label = create_subtitle_window()
subtitle_queue = queue.Queue()

# 在主线程中启动字幕更新
update_subtitle(label, subtitle_queue)  




# Use Ctrl+C to quit the program
def signal_handler(sig, frame):
    window.quit()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

window.mainloop()
