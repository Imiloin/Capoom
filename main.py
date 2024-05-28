import argparse
import queue
import signal
import sys
import threading
import time
import warnings

import numpy as np
import soundcard as sc
from soundcard import SoundcardRuntimeWarning
import tkinter as tk
import tkinter.font
from transformers import pipeline
import whisper



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
# Transcription
################################################################################

# options = whisper.DecodingOptions(task="translate")
def recognize(model):
    """
    This function is used to recognize audio and print the recognized text.
    """
    result = None
    options = whisper.DecodingOptions(without_timestamps=True)
    while True:
        audio = audio_queue.get()
        if (audio**2).max() > 0.001:
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)
            lang = max(probs, key=probs.get)
            if (not result or prev_lang != "zh") and lang == "zh":
                options = whisper.DecodingOptions(prompt="我在说简体中文哦，", without_timestamps=True)
            prev_lang = lang

            # decode the audio
            result = whisper.decode(model, mel, options)

            # print the recognized text
            print(f"{lang}: {result.text}")
            
            
            if lang == "zh":
                subtitle_zh_queue.put(result.text)
                options = whisper.DecodingOptions(task="translate", without_timestamps=True)
                result1 = whisper.decode(model, mel, options)
                subtitle_en_queue.put(result1.text)
            elif lang == "en":
                subtitle_en_queue.put(result.text)
                tobetranslated_queue.put(result.text)
            else:
                # subtitle_zh_queue.put(result.text)  # just display the original text
                options = whisper.DecodingOptions(task="translate", without_timestamps=True)
                result1 = whisper.decode(model, mel, options)
                subtitle_en_queue.put(result1.text)
                tobetranslated_queue.put(result1.text)
            
            options = whisper.DecodingOptions(prompt=result.text, without_timestamps=True)


def translate(translator):
    """
    This function is used to translate text from English to Chinese.
    """
    buffer = ""  # 创建一个缓冲区来存储短文本
    while True:
        try:
            input_text = tobetranslated_queue.get_nowait()
            if len(input_text) > 50:
                input_text = input_text[:50] + " "
        except queue.Empty:
            # 如果队列为空，暂停一会儿再检查
            time.sleep(0.1)
            continue
        
        buffer += input_text
        
        if len(buffer) < 50:  # 设置阈值为50，或一句话没有结束
            # print(buffer)
            continue
        elif buffer[-1] == "-":
            buffer[-1] == " "
            continue
        elif buffer[-3:] == "...":
            buffer = buffer[:-3] + " "
            continue

        # 翻译文本
        res = translator(buffer, max_length=500)[0]['translation_text']
        buffer = ""

        # 将翻译结果放入另一个队列
        subtitle_zh_queue.put(res)


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
            audio_queue.put(audio[:m])

            audio_prev = audio
            audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
            audio[: n - m] = audio_prev[m:n]
            n = n - m


# load the model
print("Loading model...")
model = whisper.load_model(args.model)
translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-zh")
print("Done")

# initialize the queue and the filter
audio_queue = queue.Queue()
b = np.ones(100) / 100


subtitle_en_queue = queue.Queue()
subtitle_zh_queue = queue.Queue()
tobetranslated_queue = queue.Queue()


# start recording, translating and recognizing audio in separate threads
th_recognize = threading.Thread(target=recognize, args=(model,), daemon=True)
th_recognize.start()
th_translate = threading.Thread(target=translate, args=(translator,), daemon=True)
th_translate.start()
th_record = threading.Thread(target=record, daemon=True)
th_record.start()


################################################################################
# GUI
################################################################################


def create_subtitle_window():
    window = tk.Tk()
    window.overrideredirect(True)  # 移除窗口边框
    window.attributes("-alpha", 0.7)  # 设置窗口透明度
    window.geometry("640x138+500+600")  # 设置窗口大小和位置
    window.configure(background="black")  # 设置窗口背景色
    window.attributes('-topmost', 1)  # 窗口始终保持在最前面
    
    # 检查系统中是否安装了思源黑体
    fonts = tkinter.font.families(window)
    if "思源黑体 CN" in fonts:
        font_name = "思源黑体 CN"
    else:
        font_name = "Microsoft YaHei"

    # 创建 Text 控件
    text_en = tk.Text(window, fg="white", bg="black", font=(font_name, 16), width=80, height=2, wrap="word", selectbackground="black")
    text_en.pack()
    text_zh = tk.Text(window, fg="white", bg="black", font=(font_name, 18), width=80, height=2, wrap="word", selectbackground="black")
    text_zh.pack()

    # 添加鼠标事件处理程序
    def start_move(event):
        window.x = event.x
        window.y = event.y

    def stop_move(event):
        window.x = None
        window.y = None

    def do_move(event):
        dx = event.x - window.x
        dy = event.y - window.y
        x = window.winfo_x() + dx
        y = window.winfo_y() + dy
        window.geometry(f"+{x}+{y}")

    window.bind("<ButtonPress-1>", start_move)
    window.bind("<ButtonRelease-1>", stop_move)
    window.bind("<B1-Motion>", do_move)

    return window, text_en, text_zh, font_name


def update_subtitle(text_en, text_zh, subtitle_en_queue, subtitle_zh_queue):
    try:
        subtitle_en = subtitle_en_queue.get_nowait()
        text_en.delete(1.0, tk.END)  # 清空 Text 控件
        text_en.insert(tk.END, subtitle_en)  # 插入新的字幕
        subtitle_zh = subtitle_zh_queue.get_nowait()
        text_zh.delete(1.0, tk.END)
        text_zh.insert(tk.END, subtitle_zh)
    except queue.Empty:
        pass
    text_en.after(100, update_subtitle, text_en, text_zh, subtitle_en_queue, subtitle_zh_queue)  # 每100毫秒更新一次字幕

window, text_en, text_zh, font_name = create_subtitle_window()


# 在主线程中启动字幕更新
update_subtitle(text_en, text_zh, subtitle_en_queue, subtitle_zh_queue)  




# Use Ctrl+C to quit the program
def signal_handler(sig, frame):
    window.quit()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

window.mainloop()
