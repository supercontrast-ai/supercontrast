import os
import requests
import subprocess
import tempfile

from typing import Optional


def load_audio_file(audio_file: str) -> str:
    if audio_file.startswith(("http://", "https://")):
        response = requests.get(audio_file)
        response.raise_for_status()

        # Extract the file extension from the URL
        file_extension = os.path.splitext(audio_file)[1]

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        return temp_file_path
    else:
        return audio_file


def convert_to_mp3(input_file: str, output_file: Optional[str] = None) -> str:
    """
    Convert the input audio file to MP3 format using FFmpeg.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str, optional): Path to the output MP3 file. If not provided,
                                     it will be generated based on the input file name.

    Returns:
        str: Path to the converted MP3 file.
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".mp3"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-acodec",
                "libmp3lame",
                "-b:a",
                "128k",
                output_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting file to MP3: {e.stderr}")
        return input_file
