import os


def get_protocols(protocol_path):
    data = {}
    method_data = []
    with open(protocol_path, "r") as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) >= 5:
                speaker = columns[0]
                file_name = columns[1]
                spoof_method = columns[3]
                status = columns[4]
                if spoof_method not in method_data:
                    method_data.append(spoof_method)
                data[file_name] = [speaker, file_name, '-', spoof_method, status]
    return data, method_data


def get_long_bonafide_spoof_audio(data, file_path, save_path, data_label, dataset_type):
    print(f"Processing {data_label} data for {dataset_type} set...")
    if not os.path.exists(save_path):
        print(f"Creating directory: {save_path}")
        os.makedirs(save_path)

    file_list = os.listdir(file_path)
    total = len(file_list)
    count = 0

    for filename in file_list:
        if filename[:filename.index(".")] in data:
            if data[filename[:filename.index(".")]][4] == data_label:
                cmd = f'cp {file_path}/{filename} {save_path}/{filename}'
                os.system(cmd)
        count += 1
        progress = (count / total) * 100
        print(f'  Progress: {count}/{total} ({progress:.2f}%)', end='\r', flush=True)

    print(f"\nFinished processing {data_label} data for {dataset_type} set.\n")


if __name__ == '__main__':
    dataset_types = ['dev', 'eval', 'train']
    data_labels = ['bonafide', 'spoof']
    database_path = '/home/yzy/data/MultiSiFer/'

    for data_label in data_labels:
        print(f"\n========== Processing {data_label.upper()} Data ==========")
        for dataset_type in dataset_types:
            protocol_path = (f'{database_path}LA/ASVspoof2019_LA_cm_protocols/'
                             f'ASVspoof2019.LA.cm.{dataset_type}.trl.txt') if dataset_type != 'train' else \
                            f'{database_path}LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
            
            file_path = f"{database_path}LA_silence/ASVspoof2019_LA_{dataset_type}_silence/flac"
            save_path = (f"{database_path}silence_data/{dataset_type}_set/"
                         f"{data_label}_data_silence/flac") if dataset_type != 'train' else \
                        f"{database_path}silence_data/train_set/{data_label}_data_silence/flac"

            data, _ = get_protocols(protocol_path)
            get_long_bonafide_spoof_audio(data, file_path, save_path, data_label, dataset_type)