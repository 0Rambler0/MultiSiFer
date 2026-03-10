import os
from tqdm import tqdm

def move_audio(mix_file_type):
    for type in tqdm(dataset_type, desc=f"Processing {mix_file_type}"):
        file_path = f'{database_path}silence_data/{type}_set/{mix_file_type}/flac/'
        save_path = f'{database_path}multi_speaker_LA/{type}/flac/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cmd = 'cp -r {}* {}'.format(file_path, save_path)
        os.system(cmd)
    print(f'{mix_file_type} done.')
    return

def cat_protocols(true_file, spoof_file, type):
    with open(true_file, 'r', encoding='utf-8') as file1:
        content1 = file1.read()

    with open(spoof_file, 'r', encoding='utf-8') as file2:
        content2 = file2.read()

    combined_content = str(content1) + str(content2)
    protocol_save_path = f'{database_path}multi_speaker_LA/protocols/'

    if not os.path.exists(protocol_save_path):
        os.makedirs(protocol_save_path)

    save_file_name = f'{type}.trn.txt' if type == 'train' else f'{type}.trl.txt'
    with open(f'{protocol_save_path}{save_file_name}', 'w', encoding='utf-8') as combined_file:
        combined_file.write(combined_content)

if __name__ == '__main__':
    mix_file_types = ['mix_true_spoof', 'mix_all_true']
    dataset_type = ['eval', 'dev', 'train']
    database_path = '/home/yzy/data/MultiSiFer/'

    # Move audio files with progress bars
    for mix_file_type in tqdm(mix_file_types, desc="Moving audio files"):
        move_audio(mix_file_type)

    # Concatenate protocol files with progress bars
    for type in tqdm(dataset_type, desc="Concatenating protocols"):
        true_file = f'{database_path}silence_data/{type}_set/mix_protocol/mix_all_true_protocols.txt'
        spoof_file = f'{database_path}silence_data/{type}_set/mix_protocol/mix_true_spoof_protocols.txt'
        cat_protocols(true_file, spoof_file, type)