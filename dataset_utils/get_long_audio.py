import os
import librosa

data_type = ['spoof_data', 'bonafide_data']
database_path = '/home/yzy/data/MultiSiFer/'
mix_data = os.listdir('/home/yzy/data/MultiSiFer/silence_data')


for dataset in mix_data:
    print(f"Processing dataset: {dataset}")
    for i in data_type:
        dataset_path = f'{database_path}silence_data/{dataset}/{i}_silence/flac'
        save_path = f'{database_path}silence_data/{dataset}/long_{i}_silence/flac'
        file_list = os.listdir(dataset_path)

        count_long = 0
        total_files = len(file_list)

        if not os.path.exists(save_path):
            print(f"Creating directory: {save_path}")
            os.makedirs(save_path)

        for idx, filename in enumerate(file_list, start = 1):
            file_path = os.path.join(dataset_path, filename)
            duration = librosa.get_duration(path = file_path)

            # Print progress
            progress = (idx / total_files) * 100
            print(f"Processing: {idx}/{total_files} ({progress:.2f}%) - File: {filename}", end='\r', flush=True)

            if duration > 2:
                count_long += 1
                save_file_path = os.path.join(save_path, filename)
                os.system(f'cp "{file_path}" "{save_file_path}"')

        print(f"\nFinished processing {i} for {dataset}. Long audio files: {count_long}\n")