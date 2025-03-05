# Book Highlighter

Book Highlighter takes in a a typical "Read Aloud" video of a person reading a book aloud and highlights the words as they are being spoken.  There are many read aloud children's book videos on sites like YouTube, and that was the initial target of this project.

A GIF showing a short demo of a video procssed by this project.

![BB_highlighted GIF](https://github.com/user-attachments/assets/0b75eae2-b68d-4649-9c6a-d4f68ed49908)

Full video, including audio, the GIF is extracted from.

https://github.com/user-attachments/assets/d065b693-9834-40e0-8890-4d1eb1d76f3e

Note that this is a personal project.  It gave me a way to play around with Open CV, traditional OCR, Vision APIs, and Speech-To-Text.  Eventually this lead me to play with Text-To-Speech, which was used for the demo video shown above.

# Installation 

This project is written in Python.  It also uses the openai-whisper package, which doesn't current support Python versions newer than 3.11. I've tested it with python 3.10 and 3.11, so I recommend either of those.

The project also contains a poetry pyproject.toml file, which has all the packages required for this project to run.

# Usage

Running this project is done by running the main.py script via the command line. 

Command line usage example
```
poetry run python main.py .\file_path\video_file_name
```

Save log message to app.log in \logs directory
```
poetry run python main.py .\file_path\video_file_name --log_to_file
```

After you run the script, you'll have a video file with just the words highlighted, but with no audio.  This needs to be combined with the audio track from the original audio.  Here is an example of how ffmpeg can be used to extract to this

```
ffmpeg -i {original_video} -map 0:a ./output/audio_extracted.mp3
ffmpeg -i {output_video} -i ./output/audio_extracted.mp3 -c:v copy -c:a aac ./output/video/combined.mp4
```

Since this is an early version of the project, the output paths are hardcoded.

Note that this project uses openai-whisper and EasyOCR, both of which support GPUs.  However, a GPU is not required.  Since read-aloud videos are typically short (e.g. 5-10 minutes), a relatively new CPU can highlight the entire video in a reasonable amount of time.

# To Do
## Bigger Changes
- Easy UI to Crop Frames (e.g. so watermarks aren't transcribed)
- Page skew correction
- Test and handle books where the text is on different parts of a page
## Small Changes
- Add Better error Handing to File Loads
- Add Debug Information into Video Output
- Change output paths to not be hardcoded
- Combine the final video track and the original audio track (this needs to be done manually for now)
