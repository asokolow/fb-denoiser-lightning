import torchaudio
import os
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, _, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    with tqdm(total=len(audio_files)) as pbar:
        for file in audio_files:
            fileId = os.path.basename(file).rsplit(".", 1)[0]
            siginfo, _ = torchaudio.info(file)
            length = siginfo.length // siginfo.channels
            meta.append((file, length, fileId))
            if progress:
                pbar.update(1)
    return meta


def validate_dataset(clean_d, noise_d):
    clean_sorted = sorted(clean_d, key=lambda tup: tup[-1])
    noisy_sorted = sorted(noise_d, key=lambda tup: tup[-1])
    for cl, ns in zip(clean_sorted, noisy_sorted):     
        assert cl[-1] == ns[-1] and cl[1] == ns[1]
    return clean_sorted, noisy_sorted


def sample_dataset(clean_d, noise_d, sampling=0.):
    clean_0, noisy_0, clean_1, noisy_1  = None, None, None, None
    if sampling:
        clean_0, noisy_0 = zip(*random.sample(list(zip(clean_d, noise_d)), int(len(clean_d)*sampling)))
        clean_1, noisy_1 = zip(*list(set(list(zip(clean_d, noise_d))) - set(list(zip(clean_0, noisy_0)))))
        assert len(clean_0) + len(clean_1) == len(clean_d) # assert we use the full db
        assert len(noisy_0) + len(noisy_1) == len(noise_d) # assert we use the full db

        assert not list(set(clean_0).intersection(set(clean_1))) # assert no intersection between train and val
        assert not list(set(noisy_0).intersection(set(noisy_1))) # assert no intersection between train and val

        clean_0, noisy_0 = validate_dataset(clean_0, noisy_0)
        clean_1, noisy_1 = validate_dataset(clean_1, noisy_1)
    else:
        raise ValueError('To use this function please specify a sampling > 0.')
    return (clean_0, noisy_0), (clean_1, noisy_1)


def dump_csv(clean_d, noise_d, dir, name):
    list_data = list()
    for dt in zip(clean_d, noise_d):
        cl, ns = dt
        list_data.append((cl[0], ns[0], cl[1]))
    list_set = pd.DataFrame(data=list_data, columns=["clean_wav", "noisy_wav", "n_samples"])
    list_set.to_csv(os.path.join(dir, name), index=False)


if __name__ == "__main__":
    training_val_set = [
    ('/home/asokolow/asokolow/database/valentini/16k/clean_trainset_28spk_wav',
    '/home/asokolow/asokolow/database/valentini/16k/noisy_trainset_28spk_wav'),
    ('/home/asokolow/asokolow/database/valentini/16k/clean_trainset_56spk_wav',    
    '/home/asokolow/asokolow/database/valentini/16k/noisy_trainset_56spk_wav'),
]

    test_set = [
        ('/home/asokolow/asokolow/database/valentini/16k/clean_testset_wav',
        '/home/asokolow/asokolow/database/valentini/16k/noisy_testset_wav'),
    ]

    # training_val_set
    clean_tmp = []
    noisy_tmp = []
    for dset in training_val_set:
        clean, noisy = dset
        clean_tmp += find_audio_files(clean)
        noisy_tmp += find_audio_files(noisy)

    clean_full, noisy_full = validate_dataset(clean_tmp, noisy_tmp)

    (clean_train, noisy_train), (clean_val, noisy_val) = sample_dataset(clean_full, noisy_full, sampling=0.8)

    # test_set
    clean_tmp = []
    noisy_tmp = []
    for dset in test_set:
        clean, noisy = dset
        clean_tmp += find_audio_files(clean)
        noisy_tmp += find_audio_files(noisy)

    clean_test, noisy_test = validate_dataset(clean_tmp, noisy_tmp)

    # dump to csv file
    dump_csv(clean_train, noisy_train, 'egs/valentini', 'train.csv')
    dump_csv(clean_val, noisy_val, 'egs/valentini', 'val.csv')
    dump_csv(clean_test, noisy_test, 'egs/valentini', 'test.csv')
