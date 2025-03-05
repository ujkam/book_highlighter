# poetry run python main.py BB.mp4 --log_to_file
import whisper
import pickle
import numpy as np
import cv2 as cv
from itertools import chain

# import pytesseract
# from pytesseract import Output
from skimage.metrics import structural_similarity as ssim
import re
from PIL import Image
from itertools import compress
from collections import deque
import easyocr
from loguru import logger
import sys
import os
import argparse
from datetime import datetime

from functions import *


def main(video_file_name):
    reader = easyocr.Reader(["en"])
    current_date = datetime.now().strftime("%m_%d_%Y")

    video_file_path, video_output_file_path, transcript_file_path = create_file_paths(video_file_name, current_date)

    logger.info(f"Begin Highlighting {video_file_path}")

    logger.info(f"Begin Transcribing Audio {video_file_path}")
    transcribe_result = transcribe_audio(
        video_file_path, transcript_file_path, load_previous_file=False
    )
    transcribed_words_clean, start_times, end_times = transcription_clean(
        transcribe_result
    )
    logger.info(f"End Transcribing Audio {video_file_path}")

    ocr_list = []
    timestamps = []
    frames_nums = []
    cap = cv.VideoCapture(video_file_path)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    cap_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    cap_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    logger.debug(f"Video FPS:{cap_fps}, Height:{cap_height}, Width:{cap_width}")

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(
        video_output_file_path,
        fourcc,
        np.round(cap_fps, 2),
        (int(cap_width), int(cap_height)),
    )

    font = cv.FONT_HERSHEY_SIMPLEX
    ssim_value = -100
    ssim_values = []
    search_words = []
    word_indexes = []
    frame_list = {}
    ssim_deque = deque(maxlen=5)
    page_counter = 0
    old_word_index = 0

    all_images = []
    logger.info(f"Begin Highlighting Video {video_file_path}")
    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        timestamp = cap.get(cv.CAP_PROP_POS_MSEC)
        frame_number = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        grey_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        small_image = cv.resize(grey_image, (0, 0), fx=0.2, fy=0.2)
        large_image = cv.resize(grey_image, (0, 0), fx=2, fy=2)

        img_new = Image.fromarray(grey_image[:980, :])
        # if int(timestamp % 10) == 0:
        #    word_data = pytesseract.image_to_data(img_new, output_type=Output.DICT)

        if ssim_value < 0:
            small_image_old = small_image.copy()
            
            all_images.append(img_new)
            word_data = reader.readtext(
                np.array(img_new), width_ths=0.01, height_ths=0.01
            )
           
            left, right, top, bottom, words = extract_easyocr(word_data)
            ocr_list.append((words, timestamp))
            old_word_index = 0
            ssim_stable = True

        ssim_value = ssim(small_image_old, small_image)
        ssim_values.append(ssim_value)
        ssim_deque.append(ssim_value)

        if ssim_value < 0.85:
            ssim_stable = False

        if (np.all(np.round(ssim_deque, 2) >= 0.95)) & (ssim_stable == False):
            ssim_stable = True
        
            all_images.append(img_new)
            word_data = reader.readtext(
                np.array(img_new), width_ths=0.01, height_ths=0.01
            )
           
            left, right, top, bottom, words = extract_easyocr(word_data)
            ocr_list.append((words, timestamp))
            frame_list[frame_number] = ssim_value
            page_counter = page_counter + 1
            old_word_index = 0
            frame_test = frame.copy()

        
        clean_words = [re.sub("[^A-Za-z]+", "", x.lower()) for x in words]
        remove_list = [i != "" for i in clean_words]
        clean_words_filtered = list(compress(clean_words, remove_list))
        left_filtered = list(compress(left, remove_list))
        top_filtered = list(compress(top, remove_list))
        right_filtered = list(compress(right, remove_list))
        bottom_filtered = list(compress(bottom, remove_list))

        word_index = word_search(
            transcribed_words_clean,
            clean_words_filtered,
            timestamp / 1000,
            start_times,
            end_times,
            old_word_index=old_word_index,
        )

        if word_index > old_word_index:
            old_word_index = word_index

        if word_index != -1:
            # print(word_index)
            # old_word_index = word_index
            x0 = left_filtered[word_index]
            x1 = right_filtered[word_index]
            y0 = top_filtered[word_index]
            y1 = bottom_filtered[word_index]
            buffer = 0
            cv.rectangle(
                frame,
                (x0 - buffer, y0 - buffer),
                (x1 + buffer, y1 + buffer),
                (0, 255, 0),
                3,
            )

        # cv.putText(frame,
        #         f'Time: {np.round(timestamp,2)}, SSIM: {np.round(ssim_value,2)}, Page: {page_counter}, Index:{word_index}, Old Index:{old_word_index}',
        #             (50, 50),
        #             font, 1,
        #             (255, 0, 0),
        #             2,
        #             cv.LINE_4)
        # cv.imshow('frame', frame)
        # if cv.waitKey(1) == ord('q'):
        #    break

        out.write(frame)

        small_image_old = small_image.copy()
        

    cap.release()
    out.release()
    cv.destroyAllWindows()
    logger.info(f"End Highlighting:{video_file_path}")
    logger.info(f"Output Video:{video_output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("movie_file", type=str, help="Movie path and filename")
    parser.add_argument("--log_level", type=str, help="Log Level (INFO or DEBUG)", default='DEBUG')
    parser.add_argument('--log_to_file', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    file_path = args.movie_file
    log_level = args.log_level
    
    logger.remove()
    if args.log_to_file:
        configure_logging(log_level=log_level)
    else:
        logger.add(sys.stderr, level=log_level)
    
    logger.info("Start Application")
    main(file_path)
    logger.info("End Application")
