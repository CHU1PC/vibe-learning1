import contextlib
import wave

from pydub import AudioSegment


def cut_wav(input_path, output_path, max_sec=60):
    """
    指定された秒数でWAVファイルをカットします。
    """
    with contextlib.closing(wave.open(input_path, 'rb')) as wf:
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        duration = n_frames / float(framerate)
        cut_frames = int(min(duration, max_sec) * framerate)
        wf.rewind()
        frames = wf.readframes(cut_frames)
        with wave.open(output_path, 'wb') as out_wf:
            out_wf.setnchannels(n_channels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)
            out_wf.writeframes(frames)

def convert_and_cut_audio(input_file_path, output_cut_path, max_sec=60):
    """
    各種音声ファイルをWAVに変換し、指定された秒数でカットします。
    """
    ext = input_file_path.lower().split('.')[-1]
    temp_wav_path = "temp_audio_converted.wav"

    if ext == 'wav':
        # WAVの場合は直接カット
        cut_wav(input_file_path, output_cut_path, max_sec)
    elif ext in ['mp3', 'm4a']:
        # その他の形式はWAVに変換してからカット
        audio = AudioSegment.from_file(input_file_path, format=ext)
        audio.export(temp_wav_path, format="wav")
        cut_wav(temp_wav_path, output_cut_path, max_sec)
        # 変換後の一時ファイルを削除
        import os
        os.remove(temp_wav_path)
    else:
        raise ValueError(f"Unsupported audio format: {ext}")
