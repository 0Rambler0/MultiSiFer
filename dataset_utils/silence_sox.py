import os 
from concurrent.futures import ThreadPoolExecutor

def exec(cmd):
    os.system(cmd)


dataset_types=['dev','eval','train']
database_path = '/home/yzy/data/MultiSiFer/'
for dataset_type in dataset_types:
    print(dataset_type)
    dataset_path = database_path + 'LA/ASVspoof2019_LA_{}/flac'.format(dataset_type)
    save_path = database_path + 'LA_silence/ASVspoof2019_LA_{}_silence/flac'.format(dataset_type)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pool = ThreadPoolExecutor(24)
    fname_list = os.listdir(dataset_path)
    total = len(fname_list)
    count = 0
    for fname in fname_list:
        count += 1
        fname = fname.split('.')[0]
        target_path = os.path.join(dataset_path, fname)
        target_save_path = os.path.join(save_path, fname)
        cmd3 = 'sox {}.flac {}.flac silence 1 0.00001 1% -1 0.00001 1%%'.format(target_path, target_save_path)
        pool.submit(exec, cmd3)
        os.system(cmd3)
        if count%100 == 0:
            print('processed rate: {}/{}'.format(count, total), end='\r', flush=False)
