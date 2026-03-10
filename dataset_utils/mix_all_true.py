from pydub import AudioSegment
import random
import os
from true_select import get_protocols
from pathlib import Path

dataset_type = ['eval', 'dev', 'train']
database_path = '/home/yzy/data/MultiSiFer/'

for type in dataset_type:
    print(f"\nProcessing dataset: {type}")
    real_audio_path = f"{database_path}silence_data/{type}_set/long_bonafide_data_silence/flac/"
    real_audio_list = os.listdir(real_audio_path)

    if type == 'train':
        protocol_path = f"{database_path}LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    else:
        protocol_path = f"{database_path}LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{type}.trl.txt"
    data, _ = get_protocols(protocol_path)

    save_protocol_path = f"{database_path}silence_data/{type}_set/mix_protocol/"
    save_path = f"{database_path}silence_data/{type}_set/mix_all_true/flac/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_protocol_path):
        os.makedirs(save_protocol_path)

    # Mixing Two Audio
    def mix_audios(real_audio_filename1, real_audio_filename2, output_file, data):
        real_audio1 = AudioSegment.from_file(real_audio_path + real_audio_filename1)
        real_audio2 = AudioSegment.from_file(real_audio_path + real_audio_filename2)

        if len(real_audio1) >= len(real_audio2):
            mixed_audio = real_audio1.overlay(real_audio2)
        else:
            mixed_audio = real_audio2.overlay(real_audio1)

        # Save protocol information
        with open(f"{save_protocol_path}mix_all_true_protocols.txt", "a") as fh:
            fh.write(
                f"{data[real_audio_filename1[:real_audio_filename1.index('.')]][0]}+"
                f"{data[real_audio_filename2[:real_audio_filename2.index('.')]][0]} "
                f"{output_file_path.stem} "
                f"{data[real_audio_filename1[:real_audio_filename1.index('.')]][1]}+"
                f"{data[real_audio_filename2[:real_audio_filename2.index('.')]][1]} - bonafide\n"
            )
        # Save Mixed Audio
        mixed_audio.export(output_file, format="flac")
        return

    # Set the number of fake types for each dataset
    if type == 'dev':
        num_fake_type = 1000
    elif type == 'eval':
        num_fake_type = 5000
    else:
        num_fake_type = 2000

    counter = 1
    while counter <= num_fake_type:
        # Randomly select two true audio
        real_audio_filename1 = random.choice(real_audio_list)
        real_audio_filename2 = random.choice(real_audio_list)

        real_audio1 = AudioSegment.from_file(real_audio_path + real_audio_filename1)
        real_audio2 = AudioSegment.from_file(real_audio_path + real_audio_filename2)

        min_len = min(len(real_audio1), len(real_audio2))
        max_len = max(len(real_audio1), len(real_audio2))

        # Ensure the two audios are from different speakers and length requirements are met
        if data[real_audio_filename1[:real_audio_filename1.index(".")]][0] != data[real_audio_filename2[:real_audio_filename2.index(".")]][0] and min_len >= max_len / 2:
            output_file = f"{save_path}mix_all_true_{type}_{counter}.flac"
            output_file_path = Path(output_file)
            mix_audios(real_audio_filename1, real_audio_filename2, output_file, data)

            # Print progress
            progress = (counter / num_fake_type) * 100
            print(f"Progress: {counter}/{num_fake_type} ({progress:.2f}%)", end='\r', flush=True)
            counter += 1

    print(f"\nFinished processing {type} dataset.\n")