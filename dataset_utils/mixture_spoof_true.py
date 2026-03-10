from pydub import AudioSegment
import random
import os
from true_select import get_protocols
from pathlib import Path

dataset_type = ['eval', 'dev', 'train']
database_path = '/home/yzy/data/MultiSiFer/'

for type in dataset_type:
    print(f"Processing dataset: {type}")
    
    real_audio_path = f"{database_path}silence_data/{type}_set/long_bonafide_data_silence/flac/"
    fake_audio_path = f"{database_path}silence_data/{type}_set/long_spoof_data_silence/flac/"
    save_protocol_path = f"{database_path}silence_data/{type}_set/mix_protocol/"
    save_path = f"{database_path}silence_data/{type}_set/mix_true_spoof/flac/"
    protocol_path = f"{database_path}LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{ 'train.trn' if type == 'train' else type + '.trl' }.txt"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_protocol_path):
        os.makedirs(save_protocol_path)
    data, method_data = get_protocols(protocol_path)

    if '-' in method_data:
        method_data.remove('-')

    # Group fake audio files by their spoofing method
    for method in method_data:
        globals()[method] = [
            filename for filename in os.listdir(fake_audio_path)
            if filename[:filename.index(".")] in data and data[filename[:filename.index(".")]][3] == method
        ]

    # Function to mix real and fake audio files
    def mix_audios(real_audio_filename, fake_audio_filename, output_file, data):
        real_audio = AudioSegment.from_file(f"{real_audio_path}{real_audio_filename}")
        fake_audio = AudioSegment.from_file(f"{fake_audio_path}{fake_audio_filename}")

        # Overlay the shorter audio onto the longer one
        if len(real_audio) >= len(fake_audio):
            mixed_audio = real_audio.overlay(fake_audio)
        else:
            mixed_audio = fake_audio.overlay(real_audio)

        # Save protocol information
        with open(f"{save_protocol_path}mix_true_spoof_protocols.txt", "a") as fh:
            fh.write("{} {} {} {} spoof\n".format(
                f"{data[real_audio_filename[:real_audio_filename.index('.')]][0]}+"
                f"{data[fake_audio_filename[:fake_audio_filename.index('.')]][0]}",
                output_file_path.stem,
                f"{data[real_audio_filename[:real_audio_filename.index('.')]][1]}+"
                f"{data[fake_audio_filename[:fake_audio_filename.index('.')]][1]}",
                data[fake_audio_filename[:fake_audio_filename.index('.')]][3]
            ))

        mixed_audio.export(output_file, format="flac")

    # Set the number of mixed files to generate for each dataset type
    num_fake_type = {'dev': 250, 'eval': 1000, 'train': 500}[type]

    # Generating mixed audio files
    for method in method_data:
        fake_audio_files = globals()[method]
        print(f"\nCreating method {method} for dataset {type}...")
        counter = 1
        while counter <= num_fake_type:
            # Randomly select one real and one fake audio file
            real_audio_filename = random.choice(os.listdir(real_audio_path))
            fake_audio_filename = random.choice(fake_audio_files)

            real_audio = AudioSegment.from_file(f"{real_audio_path}{real_audio_filename}")
            fake_audio = AudioSegment.from_file(f"{fake_audio_path}{fake_audio_filename}")

            min_len = min(len(real_audio), len(fake_audio))
            max_len = max(len(real_audio), len(fake_audio))

            # Check speaker mismatch and length requirements
            if (data[real_audio_filename[:real_audio_filename.index(".")]][0] != 
                data[fake_audio_filename[:fake_audio_filename.index(".")]][0] 
                and min_len >= max_len / 2):

                output_file = f"{save_path}mix_true_spoof_{type}_{method}_{counter}.flac"
                output_file_path = Path(output_file)
            
                mix_audios(real_audio_filename, fake_audio_filename, output_file, data)

                # Update progress bar
                progress = (counter / num_fake_type) * 100
                print(f"Progress: {counter}/{num_fake_type} ({progress:.2f}%)", end='\r', flush=True)

                counter += 1
        print(f"\nFinished method {method} for dataset {type}.\n")