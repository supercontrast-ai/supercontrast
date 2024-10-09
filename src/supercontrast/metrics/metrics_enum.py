from enum import Enum


class Metric(Enum):
    # Word-level metrics
    WER = "word_error_rate"
    MER = "match_error_rate"
    WIL = "word_information_lost"
    WIP = "word_information_preserved"
    WRR = "word_recognition_rate"

    # Character-level metric
    CER = "character_error_rate"

    # Translation-specific metrics
    BLEU = "bilingual_evaluation_understudy"
    BLEU_NLTK = "bleu_nltk"
    METEOR = "metric_for_evaluation_of_translation_with_explicit_ordering"
    CHRF = "character_n_gram_f_score"
