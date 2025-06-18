#!/usr/bin/env python3
# !/usr/bin/env python3
from moviepy.editor import AudioFileClip


def convert_mp4a_to_wav(input_file, output_file=None):
    if not output_file:
        output_file = input_file.rsplit(".", 1)[0] + ".wav"

    audio = AudioFileClip(input_file)
    audio.write_audiofile(output_file, fps=44100, codec="pcm_s16le")
    print(f"转换完成: {input_file} -> {output_file}")


# 示例
if __name__ == "__main__":
    convert_mp4a_to_wav("test/3331.m4a", "cs1.wav")

