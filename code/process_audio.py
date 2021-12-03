# importing packages
import os, glob
from itertools import chain
from pydub import AudioSegment

# defining path to data
raw_path = "../raw_data/"
subjects = os.listdir(raw_path)

# create new data folder
if not os.path.exists("../data/"):
    os.makedirs("../data/")

for subject in subjects:
    if not os.path.exists("../data/" + subject):
        os.makedirs("../data/" + subject)
        os.makedirs("../data/" + subject + "/read/")
        os.makedirs("../data/" + subject + "/sing/")

# getting all relevant data files
speaking_files = [glob.glob(raw_path + subject + "/read/*.wav") for subject in subjects]
speaking_files = list(chain(*speaking_files))

singing_files = [glob.glob(raw_path + subject + "/sing/*.wav") for subject in subjects]
singing_files = list(chain(*singing_files))

audio_files = speaking_files + singing_files
labels = ["speaking"] * len(speaking_files) + ["singing"] * len(singing_files)

# process raw data to PCM_16
def process_audio_file(audio_file):
    new_path = audio_file.replace("raw_data", "data")
    audio = AudioSegment.from_wav(audio_file)[2000:4000]
    audio = audio.set_frame_rate(44100)
    audio = audio.set_sample_width(2)
    audio.export(new_path, format = "wav", bitrate = "16k")
    
for audio_file in audio_files:
    process_audio_file(audio_file)