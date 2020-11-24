import argparse
import os
import shutil
import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from denoiser.demucs_lightning import DemucsLightning

if os.path.exists('./checkpoint'):
    shutil.rmtree('./checkpoint')
    os.mkdir('./checkpoint')
else:
    os.mkdir('./checkpoint')

if os.path.exists('./samples'):
    shutil.rmtree('./samples')
    os.mkdir('./samples')
else:
    os.mkdir('./samples')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # Dataset related
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--segment_sec", default=4, type=int)
    parser.add_argument("--hop_sec", default=1, type=int, help="in seconds, how much to stride between training examples")
    parser.add_argument("--pad", action="store_true", help="if training sample is too short, pad it")

    #ds default
    parser.add_argument("--train", default='egs/valentini/train.csv', help='path to train csv file')
    parser.add_argument("--valid", default='egs/valentini/val.csv', help='path to val csv file')
    parser.add_argument("--test", default='egs/valentini/test.csv', help='path to test csv file')

    # Dataset Augmentation
    parser.add_argument("--remix", action="store_true", help="remix noise and clean")
    parser.add_argument("--bandmask", default=0., type=float, help='drop at most this fraction of freqs in mel scale')
    parser.add_argument("--shift", default=0, type=int, help='random shift, number of samples')
    parser.add_argument("--shift_same", action="store_true", help="shift noise and clean by the same amount")
    parser.add_argument("--revecho", default=0, type=int, help="add reverb like augment")

    # Checkpointing, by default automatically load last checkpoint
    parser.add_argument("--samples_dir", default='samples')
    
    # Other stuff
    parser.add_argument("--seed", default=2036, type=int)
    parser.add_argument("--n_worker", default=1, type=int)
    parser.add_argument("--deploy", action="store_true", help="use only flags as info")

    # Evaluation stuff
    parser.add_argument("--pesq", action="store_true", help="compute pesq")
    parser.add_argument("--dry", default=0., type=float, help='dry/wet knob value at eval')
    parser.add_argument("--streaming", action="store_true", help="use streaming evaluation for Demucs")

    # Optimization related
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--scheduler", action="store_true", help="use scheduler")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--loss", default='l1')
    parser.add_argument("--stft_loss", action="store_true")    
    parser.add_argument("--stft_sc_factor", default=.5, type=float)
    parser.add_argument("--stft_mag_factor", default=.5, type=float)
    parser.add_argument("--batch_size", default=10, type=int)

    # Models
    parser.add_argument("--chin", default=1, type=int)
    parser.add_argument("--chout", default=1, type=int)
    parser.add_argument("--hidden", default=48, type=int)
    parser.add_argument("--max_hidden", default=10000, type=int)
    parser.add_argument("--causal", action="store_true")    
    parser.add_argument("--glu", action="store_true")    
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--kernel_size", default=8, type=int)
    parser.add_argument("--model_stride", default=4, type=int)
    parser.add_argument("--normalize", action="store_true")    
    parser.add_argument("--resample", default=4, type=int)
    parser.add_argument("--growth", default=2, type=int)
    parser.add_argument("--rescale", default=.1, type=float)
    parser.add_argument("--clip_value", default=-1.0, type=float)
    
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    experiment_name = "demucs-lightning"
    model = DemucsLightning(hparams=args)

    mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f'Size: {mb:.1f} MB')
    field = model.valid_length(1)
    print(f'Field: {field / args.sample_rate * 1000:.1f} ms or {field} samples ({args.sample_rate} Hz)')

    if args.deploy:
        # code to deploy and test on power machine
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping

        early_stopping = EarlyStopping(monitor='val_loss', verbose=True, patience=5, strict=False, mode='min')
        checkpoint_callback = ModelCheckpoint(
            dirpath='./checkpoint',
            filename='{epoch}-{val_loss:.2f}',
            monitor='val_loss')
        comet_logger = CometLogger(
        api_key="#",
        workspace="#",  # Optional
        project_name="demucs",
        )        
        trainer = pl.Trainer.from_argparse_args(args,
            logger=comet_logger,
            # callbacks=[checkpoint_callback, early_stopping],
            callbacks=[checkpoint_callback],
            )
    else:       
        # code to test locally if pipeline is ok
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(filepath='./checkpoint')
        trainer = pl.Trainer.from_argparse_args(args,
            gpus=1,
            fast_dev_run=True,
            checkpoint_callback=checkpoint_callback,
            max_epochs=2,
            weights_summary='full'
            )
    trainer.fit(model)
    chck = glob.glob("./checkpoint/*.ckpt")[0]
    trainer.test(ckpt_path=chck)    
