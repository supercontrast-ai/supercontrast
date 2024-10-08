import difflib
import gradio as gr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import re
import unicodedata

from jiwer import cer, mer, wer, wil
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import single_meteor_score
from num2words import num2words
from PIL import Image
from sacrebleu import sentence_bleu

from supercontrast.client import supercontrast_client
from supercontrast.provider import Provider
from supercontrast.task import (
    OCRRequest,
    OCRResponse,
    Task,
    TranscriptionRequest,
    TranslationRequest,
)

# initialize nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

# Constants
TEST_IMAGE_URL = "https://jeroen.github.io/images/testocr.png"
TEST_AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/master/mono_44100/127389__acclivity__thetimehascome.wav"
TEST_TEXT = "Hello, world! This is a test translation."

# Define providers
PROVIDERS = [
    Provider.AWS,
    Provider.GCP,
    Provider.AZURE,
    Provider.SENTISIGHT,
    Provider.CLARIFAI,
    Provider.API4AI,
]

OCR_PROVIDERS = ["API4AI", "AWS", "AZURE", "CLARIFAI", "GCP", "SENTISIGHT"]

# Define output directory for saving plots
OUTPUT_DIR = "test_data/ocr"


def get_image_path(image_input):
    if isinstance(image_input, str):
        if os.path.isdir(image_input):
            # If it's a directory, find the first image file
            for file in os.listdir(image_input):
                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
                ):
                    return os.path.join(image_input, file)
            raise ValueError(f"No image file found in directory: {image_input}")
        elif os.path.isfile(image_input):
            return image_input
        else:
            raise ValueError(f"Invalid image path: {image_input}")
    elif isinstance(image_input, Image.Image):
        # If it's already a PIL Image, save it temporarily and return the path
        temp_path = os.path.join(OUTPUT_DIR, "temp_input_image.png")
        image_input.save(temp_path)
        return temp_path
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def plot_bounding_boxes(
    image_path: str, responses: dict[Provider, OCRResponse], output_dir: str
):
    # Ensure we have a valid file path
    image_path = get_image_path(image_path)

    img = Image.open(image_path)

    results = {}
    for provider, ocr_response in responses.items():
        # Create a new figure for each provider
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        for box in ocr_response.bounding_boxes:
            # Create a Rectangle patch
            rect = patches.Rectangle(
                (box.coordinates[0][0], box.coordinates[0][1]),
                box.coordinates[2][0] - box.coordinates[0][0],
                box.coordinates[2][1] - box.coordinates[0][1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add text annotation
            ax.text(
                box.coordinates[0][0],
                box.coordinates[0][1],
                box.text,
                color="blue",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        ax.axis("off")

        # Remove any extra white space around the image
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Convert plot to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_result = Image.fromarray(img_array)

        # Save the plot as an image file
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"ocr_{str(provider)}_{image_name}")
        img_result.save(output_path)
        print(f"Saved {str(provider)} plot to: {output_path}")

        # Store the image and text result
        results[provider] = {"image": img_result, "text": ocr_response.all_text}

        plt.close(fig)

    return results


def process_ocr(image, providers):
    image_path = get_image_path(image)
    results = {}
    for provider in providers:
        client = supercontrast_client(task=Task.OCR, providers=[Provider[provider]])
        response = client.request(OCRRequest(image=image_path))
        results[Provider[provider]] = response

    plot_results = plot_bounding_boxes(image_path, results, OUTPUT_DIR)

    response = []
    for provider in OCR_PROVIDERS:
        if Provider[provider] in plot_results:
            response.extend(
                [
                    plot_results[Provider[provider]]["image"],
                    plot_results[Provider[provider]]["text"],
                ]
            )
        else:
            response.extend([Image.open("dummy.png"), "foo bar"])

    return response


def normalize_text(text, task="transcription"):
    # Convert to lowercase
    text = text.lower()

    # Unicode normalization (convert to standard form)
    text = unicodedata.normalize("NFKC", text)

    # Replace numbers with their word equivalents
    def replace_number(match):
        number = match.group()
        try:
            return num2words(int(number))
        except ValueError:
            return num2words(float(number))

    text = re.sub(r"\b\d+(?:\.\d+)?\b", replace_number, text)

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = " ".join(text.split())

    if task == "translation":
        # Additional normalization steps for translation
        # (e.g., handling of diacritics might differ)
        pass

    return text


def line_by_line_diff(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(),
        text2.splitlines(),
        lineterm="",
        n=0,  # This removes the headers
    )
    return "\n".join(list(diff)[2:])  # Skip the first two lines (headers)


def word_by_word_diff(text1, text2):
    words1, words2 = text1.split(), text2.split()
    seq_matcher = difflib.SequenceMatcher(None, words1, words2)
    diff = []
    for opcode, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if opcode == "equal":
            diff.extend(words1[i1:i2])
        elif opcode == "insert":
            diff.extend([f"+{word}" for word in words2[j1:j2]])
        elif opcode == "delete":
            diff.extend([f"-{word}" for word in words1[i1:i2]])
        elif opcode == "replace":
            diff.extend([f"-{word}" for word in words1[i1:i2]])
            diff.extend([f"+{word}" for word in words2[j1:j2]])
    return " ".join(diff)


def calculate_transcription_metrics(reference, hypothesis):
    # Word-level metrics
    wer_score = wer(reference, hypothesis)
    mer_score = mer(reference, hypothesis)
    wil_score = wil(reference, hypothesis)

    # Character-level metric
    cer_score = cer(reference, hypothesis)

    # Word Information Preserved (WIP)
    wip_score = 1 - wil_score

    # Word Recognition Rate (WRR)
    wrr_score = 1 - wer_score

    return f"""WER (Word Error Rate): {wer_score:.4f}
MER (Match Error Rate): {mer_score:.4f}
WIL (Word Information Lost): {wil_score:.4f}
CER (Character Error Rate): {cer_score:.4f}
WIP (Word Information Preserved): {wip_score:.4f}
WRR (Word Recognition Rate): {wrr_score:.4f}"""


def process_transcription(audio, providers, expected_transcription=None) -> list[str]:
    results = {}
    for provider in providers:
        client = supercontrast_client(
            task=Task.TRANSCRIPTION, providers=[Provider[provider]]
        )
        response = client.request(TranscriptionRequest(audio_file=audio))
        results[provider] = response.text

    if expected_transcription:
        normalized_expected = normalize_text(expected_transcription)
        for provider, text in results.items():
            normalized_text = normalize_text(text)
            line_diff = line_by_line_diff(normalized_expected, normalized_text)
            word_diff = word_by_word_diff(normalized_expected, normalized_text)
            metrics = calculate_transcription_metrics(
                normalized_expected, normalized_text
            )
            results[provider] = {
                "original": text,
                "normalized": normalized_text,
                "line_diff": line_diff if line_diff else "No differences found.",
                "word_diff": (
                    word_diff
                    if word_diff != normalized_expected
                    else "No differences found."
                ),
                "metrics": metrics,
            }

        return [
            f"Transcriptions:\n"
            f"{provider}: {results.get(provider, {}).get('original', '')}\n"
            f"Expected: {expected_transcription}\n\n"
            f"Normalized Transcriptions:\n"
            f"{provider}: {results.get(provider, {}).get('normalized', '')}\n"
            f"Expected: {normalized_expected}\n\n"
            f"Diff (Normalized, line-by-line):\n{results.get(provider, {}).get('line_diff', '')}\n\n"
            f"Diff (Normalized, word-by-word):\n{results.get(provider, {}).get('word_diff', '')}\n\n"
            f"Metrics:\n{results.get(provider, {}).get('metrics', '')}"
            for provider in providers
        ]
    else:
        return [f"{results.get(provider, '')}" for provider in ["AZURE", "OPENAI"]]


# Add this new function to calculate translation-specific metrics
def calculate_translation_metrics(reference, hypothesis):

    # Tokenize the input for METEOR score
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)

    # BLEU score (using sacrebleu)
    bleu_score = sentence_bleu(hypothesis, [reference]).score

    # BLEU score (using NLTK)
    nltk_bleu_score = nltk_sentence_bleu([reference_tokens], hypothesis_tokens)

    # METEOR score
    meteor_score = single_meteor_score(reference_tokens, hypothesis_tokens)

    # chrF score
    chrf_score = sentence_chrf(hypothesis, [reference])

    return f"""BLEU Score (sacrebleu): {bleu_score:.4f}
BLEU Score (NLTK): {nltk_bleu_score:.4f}
METEOR Score: {meteor_score:.4f}
chrF Score: {chrf_score:.4f}"""


def language_name_to_code(language_name):
    language_map = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
    }
    return language_map.get(language_name, "en")  # Default to English if not found


def process_translation(
    text, providers, source_lang, target_lang, expected_translation=None
):
    results = {}
    source_lang_code = language_name_to_code(source_lang)
    target_lang_code = language_name_to_code(target_lang)

    for provider in providers:
        client = supercontrast_client(
            task=Task.TRANSLATION,
            providers=[Provider[provider]],
            source_language=source_lang_code,
            target_language=target_lang_code,
        )
        response = client.request(TranslationRequest(text=text))
        results[provider] = response.text

    output = []
    all_providers = ["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"]

    for provider in all_providers:
        if provider in providers:
            if expected_translation:
                normalized_expected = normalize_text(
                    expected_translation, task="translation"
                )
                translation = results[provider]
                normalized_translation = normalize_text(translation, task="translation")
                line_diff = line_by_line_diff(
                    normalized_expected, normalized_translation
                )
                word_diff = word_by_word_diff(
                    normalized_expected, normalized_translation
                )
                metrics = calculate_translation_metrics(
                    normalized_expected, normalized_translation
                )

                output.append(
                    f"Translations ({source_lang} to {target_lang}):\n"
                    f"{provider}: {translation}\n"
                    f"Expected: {expected_translation}\n\n"
                    f"Normalized Translations:\n"
                    f"{provider}: {normalized_translation}\n"
                    f"Expected: {normalized_expected}\n\n"
                    f"Diff (Normalized, line-by-line):\n{line_diff}\n\n"
                    f"Diff (Normalized, word-by-word):\n{word_diff}\n\n"
                    f"Metrics:\n{metrics}"
                )
            else:
                output.append(results[provider])
        else:
            output.append("")  # Empty string for providers not selected

    return output


def gradio_demo():
    with gr.Blocks(
        css="""
        #title {
            font-size: 3em !important;
            font-weight: 600 !important;
            margin-bottom: 0.5em !important;
        }

        .gr-box > div > label {
            font-size: 1.2em !important;
            font-weight: 500 !important;
        }

        .gr-form > div > label, .gr-form > label {
            font-size: 1.2em !important;
            font-weight: 600 !important;
        }

        .gr-form > div > label[for^='component-'], .gr-form > label[for^='component-'] {
            font-size: 1.2em !important;
            font-weight: 600 !important;
        }
    """
    ) as demo:
        gr.Markdown("# SuperContrast Demo", elem_id="title")

        with gr.Tab("OCR"):
            ocr_input = gr.Image(type="filepath", label="Input Image")
            selected_providers = gr.CheckboxGroup(
                choices=OCR_PROVIDERS, label="Providers"
            )
            ocr_button = gr.Button("Process OCR")

            # Dynamic output creation
            ocr_outputs = []
            for provider in OCR_PROVIDERS:
                with gr.Row():
                    with gr.Column(visible=False) as provider_column:
                        ocr_outputs.append(gr.Image(label=f"{provider} OCR Result"))
                        ocr_outputs.append(gr.Textbox(label=f"{provider} OCR Text"))

                    selected_providers.change(
                        lambda p, prov=provider: gr.update(visible=prov in p),
                        inputs=[selected_providers],
                        outputs=[provider_column],
                    )

            ocr_button.click(
                process_ocr, inputs=[ocr_input, selected_providers], outputs=ocr_outputs
            )

        with gr.Tab("Transcription"):
            transcription_input = gr.Audio(type="filepath", label="Input Audio")
            expected_transcription = gr.Textbox(
                label="Expected Transcription (Optional)",
                placeholder="Enter expected transcription here",
            )
            transcription_providers = gr.CheckboxGroup(
                choices=["AZURE", "OPENAI"], label="Providers"
            )
            transcription_button = gr.Button("Process Transcription")

            # Dynamic output creation for transcription
            transcription_outputs = []
            for i in range(0, len(["AZURE", "OPENAI"]), 2):
                with gr.Row():
                    for provider in ["AZURE", "OPENAI"][i : i + 2]:
                        with gr.Column(visible=False) as provider_column:
                            transcription_outputs.append(
                                gr.Textbox(label=f"{provider} Transcription Result")
                            )

                        transcription_providers.change(
                            lambda p, prov=provider: gr.update(visible=prov in p),
                            inputs=[transcription_providers],
                            outputs=[provider_column],
                        )

            transcription_button.click(
                process_transcription,
                inputs=[
                    transcription_input,
                    transcription_providers,
                    expected_transcription,
                ],
                outputs=transcription_outputs,
            )

        with gr.Tab("Translation"):
            translation_input = gr.Textbox(label="Text to Translate")
            source_lang = gr.Dropdown(
                choices=["English", "Spanish", "French", "German", "Italian"],
                label="Source Language",
                value="English",
            )
            target_lang = gr.Dropdown(
                choices=["English", "Spanish", "French", "German", "Italian"],
                label="Target Language",
                value="Spanish",
            )
            expected_translation = gr.Textbox(label="Expected Translation (Optional)")
            translation_providers = gr.CheckboxGroup(
                ["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"],
                label="Providers",
            )
            translation_button = gr.Button("Translate")

            # Dynamic output creation for translation
            translation_outputs = []
            providers = ["ANTHROPIC", "AWS", "AZURE", "GCP", "MODERNMT", "OPENAI"]
            for i in range(0, len(providers), 2):
                with gr.Row():
                    for provider in providers[i : i + 2]:
                        with gr.Column(visible=False) as provider_column:
                            translation_outputs.append(
                                gr.Textbox(label=f"{provider} Translation Result")
                            )

                        translation_providers.change(
                            lambda p, prov=provider: gr.update(visible=prov in p),
                            inputs=[translation_providers],
                            outputs=[provider_column],
                        )

            translation_button.click(
                process_translation,
                inputs=[
                    translation_input,
                    translation_providers,
                    source_lang,
                    target_lang,
                    expected_translation,
                ],
                outputs=translation_outputs,
            )

    demo.launch()


if __name__ == "__main__":
    gradio_demo()
