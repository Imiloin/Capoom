import queue
import time

import whisper




def recognize(model, audio_queue, subtitle_zh_queue, subtitle_en_queue, tobetranslated_queue):
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
                options = whisper.DecodingOptions(
                    prompt="我在说简体中文哦，", without_timestamps=True
                )
            prev_lang = lang

            # decode the audio
            result = whisper.decode(model, mel, options)

            # print the recognized text
            print(f"{lang}: {result.text}")

            if lang == "zh":
                subtitle_zh_queue.put(result.text)
                options = whisper.DecodingOptions(
                    task="translate", without_timestamps=True
                )
                result1 = whisper.decode(model, mel, options)
                subtitle_en_queue.put(result1.text)
            elif lang == "en":
                subtitle_en_queue.put(result.text)
                tobetranslated_queue.put(result.text)
            else:
                # subtitle_zh_queue.put(result.text)  # just display the original text
                options = whisper.DecodingOptions(
                    task="translate", without_timestamps=True
                )
                result1 = whisper.decode(model, mel, options)
                subtitle_en_queue.put(result1.text)
                tobetranslated_queue.put(result1.text)

            options = whisper.DecodingOptions(
                prompt=result.text, without_timestamps=True
            )


def translate(translator, tobetranslated_queue, subtitle_zh_queue):
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
        res = translator(buffer, max_length=500)[0]["translation_text"]
        buffer = ""

        # 将翻译结果放入另一个队列
        subtitle_zh_queue.put(res)
