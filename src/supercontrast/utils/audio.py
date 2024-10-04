import os
import requests
import tempfile


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
