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
from paddleocr import PaddleOCR,draw_ocr
import skimage
import pytesseract
from pytesseract import Output
from itertools import compress

def create_image_zip_files(images, file_prefix, date_str):
    """
    Takes in an array of PIL image arrays and writes a zip file with the images.  
    The files in the zip file are named in array order, so I would recommend 
    having the order array be the same as the page order in the book being transcribed.

     Args:
        image: array of PIL images
        load_previous_file: the name of the file (e.g. 'book_name') for the image and zip file name 

    """
    with zipfile.ZipFile(f'../output/images/{file_prefix}_{date_str}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, img in enumerate(images):
            with BytesIO() as img_byte_arr:
                #Save image to memory instead of file
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                image_filename = f'{file_prefix}_{i+1}.jpg'
                zipf.writestr(image_filename, img_byte_arr.read())
    logger.info(f"Zip of images created: '{image_filename}'")




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
                    f"Ocr Words Index:{index}, Search Word Loc:{search_word_loc}, Length ocr words:{len(ocr_words)}"
                )
                if ((index < (len(ocr_words) - 1)) & (search_word_loc < (len(transcribed_words_np) - 1)) ):  # Make sure we don't go past the last word on the page
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
                                logger.debug(f"Matched Index:{word_index}")
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

def extract_pytesseract(data: dict):
    text = data['text']
    #left = data['left']
    #top = data['top']
    #width = data['width']
    #height = data['height']

    remove_list = [i.replace(' ','') != '' for i in text]
    text_filtered = list(compress(text,remove_list))
    left_filtered = list(compress(data['left'], remove_list))
    top_filtered = list(compress(data['top'], remove_list))
    width_filtered = list(compress(data['width'],remove_list))
    height_filtered = list(compress(data['height'], remove_list))

    return ({'text':text_filtered,
            'left':left_filtered,
            'top':top_filtered,
            'width':width_filtered,
            'height':height_filtered
    })

def calc_original_coordinates(data: dict, b_box:dict, left_border, upper_border):
    """As part of the OCR pipeline, a box with the text in it is cropped from the original image
    and OCR is performed on that cropped box.  This function returns the OCR text boxes from 
    in the cropped cordinates to the original coordinates of the uncropped original image  

    Args:
        data (dict): the word box coodinates from pytessearact 
        b_box (dict): the bounding box used for the crop
        left_border (_type_): left border (additional pixels) for the crop
        upper_border (_type_): upper border (additional pixels) for the crop

    Returns:
        _type_: _description_
    """
    result = {}
    result['x0'] = [x + b_box['x0'] - left_border for x in data['left']] 
    result['y0'] = [x + b_box['y0'] - upper_border for x in data['top']] 
    result['x1'] = [x + y + b_box['x0'] - left_border for x,y in zip(data['left'],data['width'])] 
    result['y1'] = [x + y + b_box['y0'] - upper_border for x,y in zip(data['top'],data['height'])] 
    return result


def group_text_boxes(paddle_result):
    """ This takes the result from a paddle OCR call and combines the text boxes that 
    are close to another into one a single box.

    Args:
        paddle_result (_type_): a result object from a paddle ocr call where rec=False

    Returns:
        group a list of text boxes 
    """
    #Note that in a paddle OCR result, the order of the coodinates is 
    #[upper left, upper right, lower right, lower left]
    box_dist = 0
    group = []
    current_group = 0
    for i,coord in enumerate(paddle_result[0]):
        if i > 0:
            #Get the y coordinates for the upper and lower left corders
            current_box_y = (coord[0][1], coord[3][1])
            #Check if the order of boxes top to bottom or bottom to top, and calculate distance
            #between the lower left corner of the top box, and the upper left corner of the bottom box
            # accordingly 
            if previous_box_y[0] < current_box_y[0]:
                dist = current_box_y[0] - previous_box_y[1]
            else:
                dist = previous_box_y[0] - current_box_y[1]
            
            if (dist > box_height * 2):
                current_group = current_group + 1  
        box_height = (coord[3][1] - coord[0][1])   
        #print(f'Old Y1:{old_y1}, New Y1:{coord[0][1]}, Y_dist:{box_dist}, Box_Height:{box_height}')
        #Get the y coordinates for the upper and lower left corners to compare them to the next box in the list
        previous_box_y = (coord[0][1],coord[3][1])
        group.append(current_group)
    return(group)   

def process_image(image):
    
    image_np = np.array(image)
    thresh = skimage.filters.threshold_local(image_np, 25, offset=10)
    binary = image_np < thresh
    return(binary)

def get_coordinates_paddle(paddle_result):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in paddle_result:
        x0.append(i[0][0])
        y0.append(i[0][1])
        x1.append(i[2][0])
        y1.append(i[2][1])
    
    return {'x0': min(x0),
            'y0': min(y0),
            'x1': max(x1),
            'y1': max(y1)}

def get_bounding_boxes_paddle(image):
    ocr = PaddleOCR() # need to run only once to download and load model into memory
    result = ocr.ocr(image,rec=False)
    return(result)


def extract_text(image, return_original_coordinates=True, left_border = 10, right_border = 10, upper_border = 20, lower_border = 20):
    """This extracts the text from the image and returns bounding boxes for each word.  It uses a two stage OCR pipeline, where 
    PaddleOCR is used to extrance line level boxes, and that information is then combined and fed to 
    pytessearct is used to extract word level boxes

    Args:
        image (_type_): _description_
        return_original_coordinates (bool, optional): _description_. Defaults to True.
        left_border (int, optional): border for the crop box. Defaults to 10.
        right_border (int, optional): border for the crop box. Defaults to 10.
        upper_border (int, optional): border for the crop box. Defaults to 20.
        lower_border (int, optional): border for the crop box. Defaults to 20.

    Returns:
        dict: the text and the bounding boxes for each word 
    """
    ocr_output = []
    image_np = np.array(image)
    paddle_result = get_bounding_boxes_paddle(image_np)
    #Paddle OCR returns a bounding box around each line of text, but we want
    #a box around a block of text.  This code combines the line boxes that are close together into 
    #a single box.
    bounding_box_groups = group_text_boxes(paddle_result)
    group_bbs = []
    for value in set(bounding_box_groups):
        match_idxs = [i for i, val in enumerate(bounding_box_groups) if val == value]
        single_group = paddle_result[0][min(match_idxs):(max(match_idxs)+1)]
        group_bbs.append(single_group)
    b_boxes = [get_coordinates_paddle(group) for group in group_bbs]
    #Crop each block level box and perform OCR on it.  This way pytessearct performs
    #OCR on the cropped image with mostly text, and not on the entire image
    for b_box in b_boxes:
        crop_image = image.crop((b_box['x0'] - left_border, b_box['y0']- upper_border, 
                                b_box['x1']+right_border,b_box['y1']+lower_border))
        
        
        processed_image = process_image(crop_image)
        
        ocr_result = pytesseract.image_to_data(processed_image, output_type=Output.DICT)
        pytesseract_data = extract_pytesseract(ocr_result)
        #PaddleOCR sometimes sees letters in things (e.g. clouds) where there are none.
        #This filters out the bounding boxes where there is no text 
        if len(pytesseract_data['text']) > 0:
            #The coordinates for the word level bounding boxes from pytessearct will be in reference to the 
            #cropped image, not the entire image.  So we need to update the to location in the original image
            #not the cropped one
            original_coordinates = calc_original_coordinates(pytesseract_data, b_box, left_border, upper_border)
            if return_original_coordinates:
                pytesseract_data['left'] = original_coordinates['x0']
                pytesseract_data['top'] = original_coordinates['y0']
                pytesseract_data['right'] = original_coordinates['x1']
                pytesseract_data['bottom'] = original_coordinates['y1']
            ocr_output.append(pytesseract_data)
    return ocr_output

def clean_ocr_words(word_data):
    words = word_data['text']
    left = word_data['left']
    right = word_data['right']
    top = word_data['top']
    bottom = word_data['bottom']
    clean_words = [re.sub('[^A-Za-z]+','', x.lower()) for x in words]
    remove_list = [i != '' for i in clean_words]
    clean_words_filtered = list(compress(clean_words, remove_list))
    left_filtered = list(compress(left, remove_list))
    top_filtered = list(compress(top, remove_list))
    right_filtered = list(compress(right,remove_list))
    bottom_filtered = list(compress(bottom, remove_list))
    return clean_words_filtered, left_filtered, top_filtered, right_filtered, bottom_filtered

    