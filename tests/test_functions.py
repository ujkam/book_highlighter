import pytest
import numpy as np
from bookhighlighter.functions import word_search, create_start_end_times, compare_words_in_test
from loguru import logger

#poetry run python -m unittest tests/test_functions.py 
#poetry run pytest



@pytest.fixture(autouse=True)
def setup():
    logger.disable('bookhighlighter')

def test_word_search_unique_words():

    #logger.disable(None)
    transcribed_words = ['bears', 'birthday', 'by', 'stella', 'blackstone', 'bear', 'has','blown', 'up', 'ten', 'big', 'balloons']
    ocr_words = transcribed_words.copy()
    start_times, end_times = create_start_end_times(transcribed_words)
    t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
    assert t_words_track == ocr_words_track

def test_word_search_duplicated_words():

    transcribed_words = ['the', 'stairs', 'went', 'round', 'and', 'round', 'and','down','and', 'down', 'and', 'round','and','down','and','up']
    ocr_words = transcribed_words.copy()
    start_times, end_times = create_start_end_times(transcribed_words)
    t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
    assert t_words_track == ocr_words_track


def test_word_search_duplicate_last_word():

    transcribed_words = ['bears', 'birthday', 'by', 'stella', 'blackstone','bear', 'has','blown', 'up', 'ten', 'big', 'blown']
    ocr_words = transcribed_words.copy()
    start_times, end_times = create_start_end_times(transcribed_words)
    t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
    assert t_words_track == ocr_words_track

def test_word_search_first_last_word_match():

    transcribed_words = ['bears', 'birthday', 'by', 'stella', 'blackstone','bear', 'has','blown', 'up', 'ten', 'big', 'bears']
    ocr_words = transcribed_words.copy()
    start_times, end_times = create_start_end_times(transcribed_words)
    t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
    assert t_words_track == ocr_words_track

def test_transcription_longer_than_ocr_match():

    transcribed_words = ['bears', 'birthday', 'by', 'stella', 'blackstone','bear', 'has','blown', 'up', 'ten', 'big', 'bears','cat','dog','toys']
    ocr_words = transcribed_words.copy()[:-3]
    start_times, end_times = create_start_end_times(transcribed_words)
    t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
    assert t_words_track == ocr_words_track
