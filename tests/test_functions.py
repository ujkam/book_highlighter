import unittest
import numpy as np
from bookhighlighter.functions import word_search, create_start_end_times, compare_words_in_test
from loguru import logger

#poetry run python -m unittest tests/test_functions.py 



class TestFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.disable('bookhighlighter')

    def test_word_search_unique_words(self):

        #logger.disable(None)
        transcribed_words = ['bears', 'birthday', 'by', 'stella', 'blackstone', 'bear', 'has','blown', 'up', 'ten', 'big', 'balloons']
        ocr_words = transcribed_words.copy()
        start_times, end_times = create_start_end_times(transcribed_words)
        t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
        self.assertEqual(t_words_track, ocr_words_track, 'Unique Words Test Failed')

    def test_word_search_duplicated_words(self):

        transcribed_words = ['the', 'stairs', 'went', 'round', 'and', 'round', 'and','down','and', 'down', 'and', 'round','and','down','and','up']
        ocr_words = transcribed_words.copy()
        start_times, end_times = create_start_end_times(transcribed_words)
        t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
        self.assertEqual(t_words_track, ocr_words_track, 'Duplicated Words Test Failed')


    def test_word_search_duplicate_last_word(self):

        transcribed_words = ['bears', 'birthday', 'by', 'stella', 'blackstone','bear', 'has','blown', 'up', 'ten', 'big', 'blown']
        ocr_words = transcribed_words.copy()
        start_times, end_times = create_start_end_times(transcribed_words)
        t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
        self.assertEqual(t_words_track, ocr_words_track, 'Duplicate Last Word Test Failed')

    def test_word_search_first_last_word_match(self):

        transcribed_words = ['bears', 'birthday', 'by', 'stella', 'blackstone','bear', 'has','blown', 'up', 'ten', 'big', 'bears']
        ocr_words = transcribed_words.copy()
        start_times, end_times = create_start_end_times(transcribed_words)
        t_words_track, ocr_words_track = compare_words_in_test(transcribed_words,ocr_words, start_times, end_times)
        self.assertEqual(t_words_track, ocr_words_track, 'Duplicate Last Word Test Failed')


if __name__ == '__main__':
    unittest.main()