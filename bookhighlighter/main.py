# poetry run python main.py BB.mp4 --log_to_file
import whisper
import pickle
import numpy as np
import cv2 as cv
from itertools import chain

import pytesseract
from pytesseract import Output
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity as ssim
import re
from PIL import Image
from itertools import compress
from collections import deque
#import easyocr
from loguru import logger
import sys
import os
import argparse
from datetime import datetime
import zipfile
from io import BytesIO
import subprocess

from functions import *


def main(video_file_name, page_to_image_file):
    #reader = easyocr.Reader(["en"])
    current_date = datetime.now().strftime("%m_%d_%Y")

    video_file_path, video_output_file_path, transcript_file_path = create_file_paths(video_file_name, current_date)
    

    temp_path = os.path.dirname(video_output_file_path)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
        print(f"Directory '{temp_path}' created.")
    else:
        print(f"Directory '{temp_path}' already exists.")
    
    logger.info(f"Creating Temp Directory {temp_path}")
    

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

        img_new = Image.fromarray(grey_image)

        if ssim_value < 0:
            small_image_old = small_image.copy()
            
            all_images.append(img_new)
            
            word_data = extract_text(img_new)
            if len(word_data) > 0:
                clean_words_filtered, left_filtered, top_filtered, right_filtered, bottom_filtered = clean_ocr_words(word_data)
            else: 
                clean_words_filtered = []
            
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
           
            word_data = extract_text(img_new)
            if len(word_data) > 0:
                clean_words_filtered, left_filtered, top_filtered, right_filtered, bottom_filtered = clean_ocr_words(word_data)
            else: 
                clean_words_filtered = []
            #ocr_list.append((words, timestamp))
            frame_list[frame_number] = ssim_value
            page_counter = page_counter + 1
            old_word_index = 0
            frame_test = frame.copy()

        
        

        if len(clean_words_filtered) > 0:
            ocr_list.append((word_data['text'], timestamp))
        else: 
            clean_words_filtered = []
        logger.debug(f'Page: {page_counter}, Words:{clean_words_filtered}')
        word_index = word_search(
            transcribed_words_clean,
            clean_words_filtered,
            timestamp / 1000,
            start_times,
            end_times,
            old_word_index=old_word_index,
        )
        logger.debug(f"Main Function: Word Index {word_index}, Old Word Index {old_word_index}")
        #if ((word_index > old_word_index) and (word_index - old_word_index < 3)):
        if (word_index > old_word_index):
            old_word_index = word_index

        if word_index != -1:
            # print(word_index)
            # old_word_index = word_index
            x0 = int(left_filtered[word_index])
            x1 = int(right_filtered[word_index])
            y0 = int(top_filtered[word_index])
            y1 = int(bottom_filtered[word_index])
            buffer = 5
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

    # if page_to_image_file:
    #     logger.info(f"Saving Images")
    #     for num,img in enumerate(all_images):
    #         img_fname = f'../output/images/{video_file_name}_page_{num}.jpg'
    #         img.save(img_fname)
    if page_to_image_file:
        create_image_zip_files(all_images, os.path.splitext(file_path)[0], current_date)

    create_final_video(video_file_name, video_file_path, video_output_file_path, current_date)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("movie_file", type=str, help="Movie path and filename")
    parser.add_argument("--log_level", type=str, help="Log Level (INFO or DEBUG)", default='INFO')
    parser.add_argument('--log_to_file', default=False, action=argparse.BooleanOptionalAction, help='Write Logs to file (True or False)')
    parser.add_argument('--page_to_image', default=False, action=argparse.BooleanOptionalAction, help='Writes the pages in the video as images in a zip file')
    args = parser.parse_args()
    file_path = args.movie_file
    log_level = args.log_level
    save_image = args.page_to_image
    
    logger.remove()
    if args.log_to_file:
        configure_logging(log_level=log_level)
    else:
        logger.add(sys.stderr, level=log_level)
    
    logger.info("Start Application")
    main(file_path, page_to_image_file=save_image)
    logger.info("End Application")
