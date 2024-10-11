from supercontrast import (
    Provider,
    Task,
    get_supported_providers_for_task,
    get_supported_tasks_for_provider,
)


def test_get_supported_providers_for_task():
    assert set(get_supported_providers_for_task(Task.OCR)) == {
        Provider.API4AI,
        Provider.AWS,
        Provider.AZURE,
        Provider.GCP,
        Provider.SENTISIGHT,
    }
    assert set(get_supported_providers_for_task(Task.SENTIMENT_ANALYSIS)) == {
        Provider.ANTHROPIC,
        Provider.AWS,
        Provider.AZURE,
        Provider.GCP,
        Provider.OPENAI,
    }
    assert set(get_supported_providers_for_task(Task.TRANSCRIPTION)) == {
        Provider.AZURE,
        Provider.OPENAI,
    }
    assert set(get_supported_providers_for_task(Task.TRANSLATION)) == {
        Provider.ANTHROPIC,
        Provider.AWS,
        Provider.AZURE,
        Provider.GCP,
        Provider.MODERNMT,
        Provider.OPENAI,
    }


def test_get_supported_tasks_for_provider():
    assert set(get_supported_tasks_for_provider(Provider.ANTHROPIC)) == {
        Task.SENTIMENT_ANALYSIS,
        Task.TRANSLATION,
    }
    assert set(get_supported_tasks_for_provider(Provider.API4AI)) == {Task.OCR}
    assert set(get_supported_tasks_for_provider(Provider.AWS)) == {
        Task.OCR,
        Task.SENTIMENT_ANALYSIS,
        Task.TRANSLATION,
    }
    assert set(get_supported_tasks_for_provider(Provider.AZURE)) == {
        Task.OCR,
        Task.SENTIMENT_ANALYSIS,
        Task.TRANSCRIPTION,
        Task.TRANSLATION,
    }
    # assert set(get_supported_tasks_for_provider(Provider.CLARIFAI)) == {}
    assert set(get_supported_tasks_for_provider(Provider.GCP)) == {
        Task.OCR,
        Task.SENTIMENT_ANALYSIS,
        Task.TRANSLATION,
    }
    assert set(get_supported_tasks_for_provider(Provider.MODERNMT)) == {
        Task.TRANSLATION
    }
    assert set(get_supported_tasks_for_provider(Provider.OPENAI)) == {
        Task.SENTIMENT_ANALYSIS,
        Task.TRANSCRIPTION,
        Task.TRANSLATION,
    }
    assert set(get_supported_tasks_for_provider(Provider.SENTISIGHT)) == {Task.OCR}
