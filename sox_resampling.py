import sox
import os
import shutil
import glob
from tqdm import tqdm


tfm = sox.Transformer()
tfm.rate(16000, quality='v')

src_directories = [
    '/home/asokolow/asokolow/database/valentini/DS_10283_2791/clean_testset_wav',
    '/home/asokolow/asokolow/database/valentini/DS_10283_2791/clean_trainset_28spk_wav',
    '/home/asokolow/asokolow/database/valentini/DS_10283_2791/clean_trainset_56spk_wav',
    '/home/asokolow/asokolow/database/valentini/DS_10283_2791/noisy_testset_wav',
    '/home/asokolow/asokolow/database/valentini/DS_10283_2791/noisy_trainset_28spk_wav',
    '/home/asokolow/asokolow/database/valentini/DS_10283_2791/noisy_trainset_56spk_wav',
]

tgt_directories = [
    '/home/asokolow/asokolow/database/valentini/16k/clean_testset_wav',
    '/home/asokolow/asokolow/database/valentini/16k/clean_trainset_28spk_wav',
    '/home/asokolow/asokolow/database/valentini/16k/clean_trainset_56spk_wav',
    '/home/asokolow/asokolow/database/valentini/16k/noisy_testset_wav',
    '/home/asokolow/asokolow/database/valentini/16k/noisy_trainset_28spk_wav',
    '/home/asokolow/asokolow/database/valentini/16k/noisy_trainset_56spk_wav',
]

for src_d, tgt_d in zip(src_directories, tgt_directories):
    if os.path.exists(tgt_d):
        shutil.rmtree(tgt_d)
        os.mkdir(tgt_d)
    else:
        os.mkdir(tgt_d)
    
    files = glob.glob(os.path.join(src_d,'*.wav'))
    with tqdm(total=len(files)) as pbar:
        for name in files:
            zeze = os.path.basename(name)
            filename = os.path.join(tgt_d, os.path.basename(name))
            tfm.build_file(name, filename)
            pbar.update(1)
