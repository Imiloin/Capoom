import queue
import tkinter as tk
import tkinter.font


def create_subtitle_window():
    window = tk.Tk()
    window.overrideredirect(True)  # 移除窗口边框
    window.attributes("-alpha", 0.7)  # 设置窗口透明度
    window.geometry("640x138+500+600")  # 设置窗口大小和位置
    window.configure(background="black")  # 设置窗口背景色
    window.attributes("-topmost", 1)  # 窗口始终保持在最前面

    # 检查系统中是否安装了思源黑体
    fonts = tkinter.font.families(window)
    if "思源黑体 CN" in fonts:
        font_name = "思源黑体 CN"
    else:
        font_name = "Microsoft YaHei"

    # 创建 Text 控件
    text_en = tk.Text(
        window,
        fg="white",
        bg="black",
        font=(font_name, 16),
        width=80,
        height=2,
        wrap="word",
        selectbackground="black",
    )
    text_en.pack()
    text_zh = tk.Text(
        window,
        fg="white",
        bg="black",
        font=(font_name, 18),
        width=80,
        height=2,
        wrap="word",
        selectbackground="black",
    )
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
    text_en.after(
        100, update_subtitle, text_en, text_zh, subtitle_en_queue, subtitle_zh_queue
    )  # 每100毫秒更新一次字幕
