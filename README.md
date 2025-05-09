# Book Highlighter

Book Highlighter takes in a a typical "Read Aloud" video of a person reading a book aloud and highlights the words as they are being spoken.  There are many read aloud children's book videos on sites like YouTube, and that was the initial target of this project.

A GIF showing a short demo of a video processed by this project.

![BB_highlighted GIF](https://github.com/user-attachments/assets/0b75eae2-b68d-4649-9c6a-d4f68ed49908)

Full video, including audio, the GIF is extracted from.

https://github.com/user-attachments/assets/d065b693-9834-40e0-8890-4d1eb1d76f3e

Note that this is a personal project.  It gave me a way to play around with Open CV, traditional OCR, Vision APIs, and Speech-To-Text.  I didn't want to publish someone else's read aloud video on my github page, which led me to experiment with Text-To-Speech, which was used for the demo video shown above.

# Installation 

This project is written in Python.  It also uses the openai-whisper package, which doesn't current support Python versions newer than 3.11. I've tested it with python 3.10 and 3.11, so I recommend either of those.

ffmpeg is required.

The project also contains a poetry pyproject.toml file, which has all the packages required for this project to run.


# Usage

Running this project is done by running the main.py script via the command line. 

Command line usage example, with logging to the console
```
poetry run python main.py .\file_path\video_file_name
```

Save log message to app.log in \logs directory
```
poetry run python main.py .\file_path\video_file_name --log_to_file
```

To create the final video, the program executes two shell commands to create the final video.  I've only tested this on my personal computer.  If this does not work when you run it, you can manually perform this by executing the following

1. Extract the audio track from the original file
```
ffmpeg -i {original_video_path} -map 0:a {extracted audio path}
```

2. Combine this audio track to the highlighted video
```
ffmpeg -i {highlighted_video_path} -i {extracted_audio_path} -c:v copy -c:a aac {combined_video_path}
```

The video in the "highlighted_video_path" will be in the \output\temp\{todays_date} directory.  

## Command Line Options
```
--log_level INFO|DEBUG      We only have 2 debugging levels.  INFO (Default)
--log_to_file               Write Logs to File (True/False)
--page_to_image            This creates a zip file of the individual pages in the video (True/False)
```

Since this is an early version of the project, the output paths are hardcoded.

Note that this project uses the openai-whisper and PaddleOCR libraries, both of which support GPUs.  However, a GPU is not required.  Since read-aloud videos are typically short (e.g. 5-10 minutes), a relatively new CPU can highlight the entire video in a reasonable amount of time.

# To Do
## Bigger Changes
- Page skew correction
- Test and handle books where the text is on different parts of a page
## Small Changes
- Add Better error Handing to File Loads
- Add Debug Information into Video Output
- Change output paths to not be hardcoded
- Combine the final video track and the original audio track (this needs to be done manually for now)

# Release Notes
- 2025-04-04: Initial release.  Focused on getting the word highlighting functionality working, along with releasing something
- 2025-05-06: Completely revamp the OCR pipeline.  Text detection is much more accurate  
