import numpy as np
from loguru import logger
from itertools import chain
import re
import pickle
import sys
import whisper
import os
from io import BytesIO
import zipfile

def create_image_zip_files(images, file_prefix):
    with zipfile.ZipFile(f'../output/images/{file_prefix}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, img in enumerate(images):
            with BytesIO() as img_byte_arr:
                #Save image to memory
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                image_filename = f'{file_prefix}_{i+1}.jpg'
                zipf.writestr(image_filename, img_byte_arr.read())



def create_file_paths(video_file_name, date):
    video_file_folder = "../data/videos/"
    video_file_path = video_file_folder + video_file_name

    if os.path.exists(video_file_path):
        logger.info(f"File '{video_file_path}' exists.")
    else:
        logger.info(f"File '{video_file_path}' does not exist.")

    video_output_folder = "../output/video/"
    video_output_file_path = video_output_folder + video_file_name.replace(
        ".mp4", f"_words_highlighted_{date}.mp4"
    )

    transcript_file_name = video_file_name.replace(".mp4", "_transcription.pickle")
    transcript_file_folder = "../output/transcriptions/"
    transcript_file_path = transcript_file_folder + transcript_file_name

    return video_file_path, video_output_file_path, transcript_file_path




def transcribe_audio(video_file_path, transcript_file_path, load_previous_file=True):
    """
    Transcribe Audio Track using WhisperAI

    Args:
        video_file_name: 'The name and path of the video'
        load_previous_file: 'Checks if a previous transcription already exists and loads it'

    Returns:
     transcribed_result: transcription with timestamps
    """
    if load_previous_file:
        logger.info("Loading Transcription:{transcript_file_path}")
        with open(transcript_file_path, "rb") as file:
            transcribe_result = pickle.load(file)
    else:
        logger.info("Starting Transcription")
        model = whisper.load_model("medium")
        transcribe_result = model.transcribe(video_file_path, word_timestamps=True)
        with open(transcript_file_path, "wb") as file:
            pickle.dump(transcribe_result, file)
        logger.info("Wrote Transcription:{transcript_file_path}")
    return transcribe_result


def transcription_clean(data):
    """
    Format and clean up the words from an Whisper Transcription

    Args:
        data: Transcription output (including timestamps) from WhisperAI

    Returns:
     transcribed_words_clean: single list of transcribed words
     start_times: list of start times for each word
     end_times: list of end times for each word
    """

    word_segments = [x["words"] for x in data["segments"]]
    all_word_segments = chain.from_iterable(word_segments)

    time_words = [
        (np.round(x["start"], 1), np.round(x["end"], 1), x["word"])
        for x in all_word_segments
    ]
    start_times, end_times, words = list(zip(*time_words))
    start_times = np.array(start_times)
    end_times = np.array(end_times)
    transcribed_words = np.char.strip(words)
    transcribed_words_clean = [
        re.sub("[^A-Za-z0-9]+", "", x.lower()) for x in transcribed_words
    ]
    return (transcribed_words_clean, start_times, end_times)
    # transcribed_words_clean_np = np.array(transcribed_words_clean)


def extract_easyocr(data):
    """
    Format and clean up the words from an EasyOCR readtext object

    Args:
        data: OCR object from EasyOCR

    Returns:
     x0_list, x1_list, y0_list, y1_list: lists with bounding box coordinates of each word
     word_list: list of ocr'ed words
    """
    x0_list = []
    x1_list = []
    y0_list = []
    y1_list = []
    word_list = []

    for i in data:
        # I noticed that a lot of the weird OCRed words have floating point bounding coordinates.
        # So just filtering those out
        if int(i[0][0][0]) == i[0][0][0]:
            x0, y0 = i[0][0]
            x1, y1 = i[0][2]
            x0_list.append(x0)
            x1_list.append(x1)
            y0_list.append(y0)
            y1_list.append(y1)
            word_list.append(i[1])

    return x0_list, x1_list, y0_list, y1_list, word_list


def word_search(
    transcribed_words,
    ocr_words,
    timestamp,
    start_times,
    end_times,
    old_word_index,
    print_output=False,
):
    """Find the index of the element in the ocr_words list based on the search
      word in the transcribed_words list.

    Args:
        transcribed_words:  list of words from the voice transcription
        ocr_words:  list of words from the ocr of the book
        timestamp: The timestamp used to search for the word in transcribed_words
        start_times: list of timestamps indicating the when the a is spoken in the transcribed_words list
        end_times: list of timestamps indicating the point after a word is spoken in the transcribed_words list
        old_word_index: used to limit the words searched in ocr_words
        print_output: used for debugging

    Returns:
        The index of the found word, -1 otherwise.
    """

    transcribed_words_np = np.array(transcribed_words)
    ocr_words_np = np.array(ocr_words)
    output = ""
    search_word_loc = np.ravel(
        np.where(((start_times < timestamp) & (end_times > timestamp)))
    )
    search_word = transcribed_words_np[search_word_loc]
    logger.debug(f"Search Word:{search_word}")
    # Check if there is a transcribed word
    if len(search_word) > 0:
        # Find all words that match the transcribed word
        ocr_words_index = np.ravel(np.where(ocr_words_np == search_word))
        # This codepath is followed if more than one word on the page matches the transcribed word
        if len(ocr_words_index) > 1:
            logger.debug(f"Timestamp:{timestamp}, Old Index Value:{old_word_index}")
            logger.debug(f"All Ocr Index Matches: {ocr_words_index}")
            for index in ocr_words_index:
                logger.debug(
                    f"Ocr Words Index:{index}, Search Word Loc:{search_word_loc}"
                )
                if ((index < (len(ocr_words) - 1)) & (search_word_loc < (len(ocr_words) - 1)) ):  # Make sure we don't go past the last word on the page
                    logger.debug("Checking Next Word")
                    if (
                        ocr_words_np[index + 1]
                        == transcribed_words_np[search_word_loc + 1]
                    ):
                        # print(ocr_words_np[index], ocr_words_np[index+1])
                        if "word_index" not in locals():
                            # print('Checking word_index exists')
                            if index >= old_word_index:
                                word_index = index
                                # print(f'word_index assigned: {word_index}')
                                if output != ocr_words[word_index]:
                                    # print(ocr_words_np[index + 1], transcribed_words_np[search_word_loc + 1])
                                    output = ocr_words[word_index]
                                    if print_output:
                                        print(output)
                else:  # This code is run when we reach the last word on the page
                    if (
                        ocr_words_np[index - 1]
                        == transcribed_words_np[search_word_loc - 1]
                    ):
                        logger.debug("Checking Previous Word")
                        # Check to make sure the word_index wasn't already found in the last step
                        # This can happen if multiple words match,
                        if "word_index" not in locals():
                            word_index = index
                            # print(word_index)
                            if output != ocr_words[word_index]:
                                # print(ocr_words_np[index + 1], transcribed_words_np[search_word_loc + 1])
                                output = ocr_words[word_index]
                                if print_output:
                                    print(output)
            # print(word_matches)
        else:  # This code is run when there is no or only a single match of the transcribed word on the page
            logger.debug("Checking No or Single Match")
            if len(ocr_words_index) > 0:
                word_index = int(ocr_words_index)
                if output != ocr_words[word_index]:
                    output = ocr_words[word_index]
                    if print_output:
                        print(output)
    if output == "":
        return -1
    else:
        return word_index


def create_start_end_times(words):
    increment = 0.2
    start_times = np.linspace(0, increment * (len(words) - 1), num=len(words))
    end_times = np.append(start_times[1:], (np.round(start_times[-1] + increment, 2)))
    return start_times, end_times


def compare_words_in_test(transcribed_words, ocr_words, start_times, end_times):
    old_word_index = 0
    t_words_track = []
    ocr_words_track = []

    for value, item in enumerate(transcribed_words):
        timestamp = np.round((start_times[value] + end_times[value]) / 2, 1)
        word_index = word_search(
            transcribed_words,
            ocr_words,
            timestamp,
            start_times,
            end_times,
            old_word_index=old_word_index,
            print_output=False,
        )
        # print(word_index)
        if word_index > -1:
            old_word_index = word_index
        t_words_track.append(value)
        ocr_words_track.append(word_index)

    return (t_words_track, ocr_words_track)


def configure_logging(log_level):
    """
    Configure logging

    Args:
        log_level (str): The log level set for the logger.
    """
    logger.remove()
    logger.add(
        "../logs/app.log",
        format="{time} | {level} | {message}",
        rotation="10MB",
        retention="1 week",
        level=log_level,
    )
