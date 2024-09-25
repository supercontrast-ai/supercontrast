from abc import ABC, abstractmethod
from enum import Enum
import logging

from supercontrast.tasks.task_types import Task

class ProviderType(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"

class Provider(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def request(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")
    
    # @abstractmethod
    # async def request_async(self, *args, **kwargs):
    #     raise NotImplementedError("Subclass must implement abstract method")
    
    # @abstractmethod
    # def batch_request(self, *args, **kwargs):
    #     raise NotImplementedError("Subclass must implement abstract method")
    
    # @abstractmethod
    # def get_name(self) -> str:
    #     raise NotImplementedError("Subclass must implement abstract method")
    

def provider_factory(task: Task, provider: ProviderType, **config):
    # Import here to avoid circular import
    from supercontrast.providers.aws_provider import AWSOCR, AWSSentimentAnalysis, AWSTranslate
    from supercontrast.providers.azure_provider import AzureOCR, AzureSentimentAnalysis, AzureTranslation
    from supercontrast.providers.gcp_provider import GCPOCR, GCPSentimentAnalysis, GCPTranslation

    if task == Task.SENTIMENT_ANALYSIS:
        if provider == ProviderType.AWS:
            return AWSSentimentAnalysis.init_from_env()
        elif provider == ProviderType.GCP:
            return GCPSentimentAnalysis.init_from_env()
        elif provider == ProviderType.AZURE:
            return AzureSentimentAnalysis.init_from_env()
    elif task == Task.TRANSLATION:
        source_language = config.get("source_language", "en")
        target_language = config.get("target_language", "es")
        logging.info(f"Defaulting to {source_language} and {target_language} for translation")

        if provider == ProviderType.AWS:
            return AWSTranslate.init_from_env(source_language=source_language, target_language=target_language)
        elif provider == ProviderType.GCP:
            return GCPTranslation.init_from_env(source_language=source_language, target_language=target_language)
        elif provider == ProviderType.AZURE:
            return AzureTranslation.init_from_env(source_language=source_language, target_language=target_language)

    elif task == Task.OCR:
        if provider == ProviderType.AWS:
            return AWSOCR.init_from_env()
        elif provider == ProviderType.GCP:
            return GCPOCR.init_from_env()
        elif provider == ProviderType.AZURE:
            return AzureOCR.init_from_env()
