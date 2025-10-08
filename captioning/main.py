# OpenAI Whisper Speech-to-Text System
# Hindi | Bengali | Marathi
# Uses OpenAI API (Online) - Not Local Processing

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import time
from datetime import datetime
import json
import wave
import pyaudio
import tempfile
import os
import sys
from pathlib import Path
import numpy as np

# OpenAI client
try:
    from openai import OpenAI
except ImportError:
    print("Please install OpenAI: pip install openai")
    sys.exit(1)


class WhisperSpeechRecognizer:
    """
    Speech recognition using OpenAI Whisper API (Online)
    """

    def __init__(self, api_key=None):
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)

        # Audio recording parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper works best with 16kHz
        self.record_seconds = 5

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Recording state
        self.is_recording = False
        self.frames = []

        # Language settings
        self.current_language = 'hi'

        # Silence detection parameters
        self.silence_threshold = 500
        self.silence_duration = 2  # seconds

    def set_language(self, language_code):
        self.current_language = language_code

    def record_audio_chunk(self, duration=None):
        if duration is None:
            duration = self.record_seconds

        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        silence_counter = 0
        max_silence_chunks = int(self.silence_duration * self.rate / self.chunk)

        for _ in range(0, int(self.rate / self.chunk * duration)):
            if not self.is_recording:
                break

            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)

            # Silence detection
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.sqrt(np.mean(audio_data ** 2))

            if volume < self.silence_threshold:
                silence_counter += 1
                if silence_counter > max_silence_chunks:
                    break
            else:
                silence_counter = 0

        stream.stop_stream()
        stream.close()

        return frames

    def save_audio_to_file(self, frames, filename=None):
        if not filename:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            filename = temp_file.name
            temp_file.close()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return filename

    def transcribe_audio_file(self, audio_file_path):
        try:
            with open(audio_file_path, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.current_language
                )

            if os.path.exists(audio_file_path) and 'temp' in audio_file_path:
                os.remove(audio_file_path)

            return response.text
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def continuous_transcription(self, callback):
        while self.is_recording:
            frames = self.record_audio_chunk()

            if frames and len(frames) > 10:
                audio_file = self.save_audio_to_file(frames)
                text = self.transcribe_audio_file(audio_file)

                if text and text.strip():
                    callback(text)

            time.sleep(0.1)

    def cleanup(self):
        self.audio.terminate()


class WhisperGUI:
    """
    GUI Application for OpenAI Whisper Speech-to-Text
    """

    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ OpenAI Whisper Speech Recognition - Hindi | Bengali | Marathi")
        self.root.geometry("950x750")

        self.api_key = None
        self.recognizer = None

        self.is_listening = False
        self.transcription_history = []
        self.result_queue = queue.Queue()

        self.languages = {
            'hi': {'name': 'हिंदी (Hindi)', 'display_name': 'Hindi', 'font': ('Noto Sans Devanagari', 12)},
            'bn': {'name': 'বাংলা (Bengali)', 'display_name': 'Bengali', 'font': ('Noto Sans Bengali', 12)},
            'mr': {'name': 'मराठी (Marathi)', 'display_name': 'Marathi', 'font': ('Noto Sans Devanagari', 12)}
        }

        self.current_language = 'hi'

        self.default_font = ('Arial Unicode MS', 11)
        self.title_font = ('Arial', 13, 'bold')

        self.setup_gui()

        # Instead of thread loop, use Tkinter-safe after() loop
        self.root.after(100, self.update_display)

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # API Key Section
        api_frame = ttk.LabelFrame(main_frame, text="OpenAI API Configuration", padding="10")
        api_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W)

        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show='*')
        self.api_key_entry.grid(row=0, column=1, padx=10)

        self.api_connect_btn = ttk.Button(api_frame, text="🔐 Connect", command=self.connect_api)
        self.api_connect_btn.grid(row=0, column=2, padx=5)

        self.api_status = ttk.Label(api_frame, text="⚠️ Not connected", foreground='orange')
        self.api_status.grid(row=0, column=3, padx=10)

        ttk.Label(api_frame, text="Get your API key from: platform.openai.com/api-keys", font=('Arial', 9))\
            .grid(row=1, column=0, columnspan=4, pady=(5, 0))

        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(control_frame, text="Language:").grid(row=0, column=0, sticky=tk.W)

        self.language_var = tk.StringVar(value='hi')
        self.language_combo = ttk.Combobox(
            control_frame,
            textvariable=self.language_var,
            values=[lang['name'] for lang in self.languages.values()],
            state='readonly',
            width=25
        )
        self.language_combo.grid(row=0, column=1, padx=10)
        self.language_combo.current(0)
        self.language_combo.bind('<<ComboboxSelected>>', self.change_language)

        self.record_btn = ttk.Button(control_frame, text="🎤 Start Recording",
                                     command=self.toggle_recording, state='disabled', width=20)
        self.record_btn.grid(row=0, column=2, padx=10)

        ttk.Label(control_frame, text="Mode:").grid(row=0, column=3, padx=(20, 5))
        self.mode_var = tk.StringVar(value='continuous')
        mode_combo = ttk.Combobox(control_frame, textvariable=self.mode_var,
                                  values=['Continuous', 'Push-to-Talk'], state='readonly', width=15)
        mode_combo.grid(row=0, column=4)
        mode_combo.current(0)

        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        self.status_label = ttk.Label(status_frame, text="⚡ Ready - Connect API to begin", font=('Arial', 10))
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        self.cost_label = ttk.Label(status_frame, text="Cost: $0.000", font=('Arial', 9))
        self.cost_label.grid(row=0, column=2, padx=(20, 0))

        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="Live Transcription")

        self.text_display = scrolledtext.ScrolledText(
            realtime_frame, wrap=tk.WORD, width=90, height=22,
            font=self.default_font, bg='#f9f9f9'
        )
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text_display.tag_configure('timestamp', foreground='#0066cc', font=('Arial', 10))
        self.text_display.tag_configure('language', foreground='#009900', font=('Arial', 10, 'bold'))
        self.text_display.tag_configure('hindi', font=('Noto Sans Devanagari', 13))
        self.text_display.tag_configure('bengali', font=('Noto Sans Bengali', 13))
        self.text_display.tag_configure('marathi', font=('Noto Sans Devanagari', 13))

        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text="Full History")

        self.history_display = scrolledtext.ScrolledText(
            history_frame, wrap=tk.WORD, width=90, height=22,
            font=self.default_font, bg='#f5f5f5'
        )
        self.history_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)

        ttk.Button(bottom_frame, text="💾 Save Transcript", command=self.save_transcription).grid(row=0, column=0, padx=5)
        ttk.Button(bottom_frame, text="📋 Copy Text", command=self.copy_to_clipboard).grid(row=0, column=1, padx=5)
        ttk.Button(bottom_frame, text="🗑️ Clear All", command=self.clear_all).grid(row=0, column=2, padx=5)

        self.stats_label = ttk.Label(bottom_frame, text="Words: 0 | Characters: 0 | Duration: 0s", font=('Arial', 9))
        self.stats_label.grid(row=0, column=5, padx=(30, 0))

    def connect_api(self):
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("API Key Required", "Please enter your OpenAI API key")
            return

        try:
            self.recognizer = WhisperSpeechRecognizer(api_key=api_key)
            self.recognizer.set_language(self.current_language)

            self.api_status.config(text="✅ Connected", foreground='green')
            self.record_btn.config(state='normal')
            self.status_label.config(text="✅ Ready to record - Click 'Start Recording'")

            self.api_key = api_key
            messagebox.showinfo("Success", "Successfully connected to OpenAI Whisper API!")
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")
            self.api_status.config(text="❌ Connection failed", foreground='red')

    def change_language(self, event=None):
        selected_index = self.language_combo.current()
        lang_codes = list(self.languages.keys())
        self.current_language = lang_codes[selected_index]

        if self.recognizer:
            self.recognizer.set_language(self.current_language)

        self.status_label.config(
            text=f"Language set to {self.languages[self.current_language]['display_name']}"
        )

    def toggle_recording(self):
        if not self.is_listening:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.recognizer:
            messagebox.showwarning("Not Connected", "Please connect to API first")
            return

        self.is_listening = True
        self.recognizer.is_recording = True

        self.record_btn.config(text="⏹️ Stop Recording")
        self.status_label.config(text="🔴 Recording... Speak now!")

        self.record_thread = threading.Thread(target=self.recording_worker, daemon=True)
        self.record_thread.start()

    def stop_recording(self):
        self.is_listening = False
        if self.recognizer:
            self.recognizer.is_recording = False

        self.record_btn.config(text="🎤 Start Recording")
        self.status_label.config(text="✅ Ready")

    def recording_worker(self):
        def transcription_callback(text):
            if text and text.strip():
                result = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'language': self.languages[self.current_language]['display_name'],
                    'lang_code': self.current_language,
                    'text': text.strip()
                }
                self.result_queue.put(result)
                self.transcription_history.append(result)
                self.update_cost_estimate()

        self.recognizer.continuous_transcription(transcription_callback)

    def update_display(self):
        if not self.result_queue.empty():
            result = self.result_queue.get()

            self.text_display.insert(tk.END, f"[{result['timestamp']}] ", 'timestamp')
            self.text_display.insert(tk.END, f"{result['language']}: ", 'language')

            lang_tag = result['lang_code']
            if lang_tag in ['hi', 'mr']:
                tag = 'hindi'
            elif lang_tag == 'bn':
                tag = 'bengali'
            else:
                tag = None

            text_to_insert = f"{result['text']}\n\n"
            if tag:
                self.text_display.insert(tk.END, text_to_insert, tag)
            else:
                self.text_display.insert(tk.END, text_to_insert)

            self.text_display.see(tk.END)

            self.history_display.insert(
                tk.END,
                f"[{result['date']} {result['timestamp']}] {result['language']}: {result['text']}\n\n"
            )
            self.history_display.see(tk.END)

            self.update_statistics()

        self.root.after(100, self.update_display)

    def update_statistics(self):
        all_text = ' '.join([r['text'] for r in self.transcription_history])
        words = len(all_text.split())
        chars = len(all_text)
        duration = len(self.transcription_history) * 5

        self.stats_label.config(
            text=f"Words: {words} | Characters: {chars} | Duration: ~{duration}s"
        )

    def update_cost_estimate(self):
        minutes = (len(self.transcription_history) * 5) / 60
        cost = minutes * 0.006
        self.cost_label.config(text=f"Est. Cost: ${cost:.4f}")

    def save_transcription(self):
        if not self.transcription_history:
            messagebox.showwarning("No Data", "No transcription to save!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"),
                       ("Markdown files", "*.md"), ("All files", "*.*")],
            initialfile=f"whisper_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(self.transcription_history, f, ensure_ascii=False, indent=2)
                elif filename.endswith('.md'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("# Speech Transcription\n\n")
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        for item in self.transcription_history:
                            f.write(f"**[{item['timestamp']}] {item['language']}:**\n")
                            f.write(f"{item['text']}\n\n")
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        for item in self.transcription_history:
                            f.write(f"[{item['date']} {item['timestamp']}] {item['language']}: {item['text']}\n\n")

                messagebox.showinfo("Success", f"Transcription saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {str(e)}")

    def copy_to_clipboard(self):
        text = self.text_display.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Success", "Transcription copied to clipboard!")
        else:
            messagebox.showwarning("No Data", "No text to copy!")

    def clear_all(self):
        if self.transcription_history:
            if messagebox.askyesno("Confirm", "Clear all transcriptions?"):
                self.text_display.delete(1.0, tk.END)
                self.history_display.delete(1.0, tk.END)
                self.transcription_history.clear()
                self.update_statistics()
                self.cost_label.config(text="Cost: $0.000")
        else:
            messagebox.showinfo("Info", "No transcriptions to clear")


if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperGUI(root)

    def on_closing():
        if app.is_listening:
            app.stop_recording()
        if app.recognizer:
            app.recognizer.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
