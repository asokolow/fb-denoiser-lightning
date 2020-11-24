import os
import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from argparse import Namespace
from pesq import pesq
from pystoi import stoi
from torch.utils.data import Dataset, DataLoader
from denoiser.resample import upsample2, downsample2
from denoiser.stft_loss import MultiResolutionSTFTLoss
from denoiser.augment import BandMask, Remix, RevEcho, Shift


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        for _, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            if self.sample_rate is not None:
                if sr != self.sample_rate:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{self.sample_rate}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out


class NoisyCleanSet(Dataset):
    def __init__(self, csv_files, length=None, stride=None,
                 pad=True, sample_rate=None):
        """__init__.

        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        dataset = pd.read_csv(csv_files)
        clean_wav = list(dataset['clean_wav'])
        noisy_wav = list(dataset['noisy_wav'])
        n_samples = list(dataset['n_samples'])
        
        clean = [(x, y) for x, y in zip(clean_wav, n_samples)]
        noisy = [(x, y) for x, y in zip(noisy_wav, n_samples)]       

        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate, 'with_path': True}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class DemucsLightning(pl.LightningModule):

    def __init__(self, hparams, floor=1e-3):
        super(DemucsLightning, self).__init__()

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        # hparam
        self.hparams = hparams
        self.floor = floor

        if self.hparams.resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        # activation function
        self.activation = nn.GLU(1) if self.hparams.glu else nn.ReLU()

        # channel scale
        self.ch_scale = 2 if self.hparams.glu else 1

        # Model arch
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        chin = self.hparams.chin
        chout = self.hparams.chout
        hidden = self.hparams.hidden

        for index in range(self.hparams.depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, self.hparams.kernel_size, self.hparams.model_stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * self.ch_scale, 1), self.activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, self.ch_scale * hidden, 1), self.activation,
                nn.ConvTranspose1d(hidden, chout, self.hparams.kernel_size, self.hparams.model_stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(self.hparams.growth * hidden), self.hparams.max_hidden)
        
        self.lstm = BLSTM(chin, bi=not self.hparams.causal)
        if self.hparams.rescale:
            rescale_module(self, reference=self.hparams.rescale)

        # dataset
        self.training_dataset = None
        self.val_dataset = None
        self.test_dataset = None      
        self.augment_transform = None  

        # criterion 
        self.loss = None
        if self.hparams.loss == 'l1':
            self.loss = F.l1_loss
        elif self.hparams.loss == 'l2':
            self.loss = F.mse_loss
        elif self.hparams.loss == 'huber':
            self.loss = F.smooth_l1_loss
        else:
            raise ValueError(f"Invalid loss {self.hparams.loss}")

        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=self.hparams.stft_sc_factor, factor_mag=self.hparams.stft_mag_factor)
        
    
    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.hparams.resample) # upsampling de buffer d'entrée
        # cascade d'encoder
        for idx in range(self.hparams.depth):
            length = math.ceil((length - self.hparams.kernel_size) / self.hparams.model_stride) + 1
            length = max(length, 1)
        # cascade de decoder
        for idx in range(self.hparams.depth):
            length = (length - 1) * self.hparams.model_stride + self.hparams.kernel_size
        # downsampling final
        length = int(math.ceil(length / self.hparams.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.hparams.model_stride ** self.hparams.depth // self.hparams.resample

    @property
    def batch_size(self):
        return self.hparams.batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size):
        self.hparams.batch_size = batch_size

    @property
    def learning_rate(self):
        return self.hparams.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.hparams.learning_rate = learning_rate

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.hparams.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.hparams.resample == 2:
            x = upsample2(x)
        elif self.hparams.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.hparams.resample == 2:
            x = downsample2(x)
        elif self.hparams.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x

    def training_step(self, batch, batch_idx):
        (noisy, noisy_name), (clean, clean_name) = [x for x in batch]

        # data augmentation
        sources = torch.stack([noisy - clean, clean])
        sources = self.augment_transform(sources)
        noise, clean = sources
        noisy = noise + clean
        estimate = self(noisy)

        loss = self.loss(clean, estimate)
        if self.hparams.stft_loss:
            sc_loss, mag_loss = self.mrstftloss(estimate.squeeze(1), clean.squeeze(1))
            loss += sc_loss + mag_loss
        
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def _run_metrics(self, clean, estimate):
        estimate = estimate.cpu().numpy()[:, 0]
        clean = clean.cpu().numpy()[:, 0]
        if self.hparams.pesq:
            pesq_i = get_pesq(clean, estimate, sr=self.hparams.sample_rate)
        else:
            pesq_i = 0
        stoi_i = get_stoi(clean, estimate, sr=self.hparams.sample_rate)
        return pesq_i, stoi_i

    def validation_step(self, batch, batch_idx):
        (noisy, noisy_name), (clean, clean_name) = [x for x in batch]
        estimate = self(noisy)

        val_loss = self.loss(clean, estimate)
        if self.hparams.stft_loss:
            sc_loss, mag_loss = self.mrstftloss(estimate.squeeze(1), clean.squeeze(1))
            val_loss += sc_loss + mag_loss
        
        if self.hparams.streaming:
            ## not implemented yet
            ## old code
            # streamer = DemucsStreamer(model, dry=self.hparams.dry)
            # with torch.no_grad():
            #     estimate = torch.cat([
            #         streamer.feed(noisy[0]),
            #         streamer.flush()], dim=1)[None]
            pesq_i, stoi_i = None, None
        else:
            estimate = (1 - self.hparams.dry) * estimate + self.hparams.dry * noisy
            pesq_i, stoi_i = self._run_metrics(clean, estimate)

        return {'val_loss': val_loss, 'pesq_i': pesq_i, 'stoi_i': stoi_i}

    def test_step(self, batch, batch_idx):
        (noisy, noisy_name), (clean, clean_name) = [x for x in batch]
        estimate = self(noisy)
        estimate = (1 - self.hparams.dry) * estimate + self.hparams.dry * noisy
        pesq_i, stoi_i = self._run_metrics(clean, estimate)

        filename = os.path.join(self.hparams.samples_dir, os.path.basename(noisy_name[0]).rsplit(".", 1)[0])
        write(noisy.squeeze(0), filename + "_noisy.wav", sr=self.hparams.sample_rate)
        write(estimate.squeeze(0), filename + "_enhanced.wav", sr=self.hparams.sample_rate)

        return {'pesq_i': pesq_i, 'stoi_i': stoi_i}
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        pesq_i_mean = torch.stack([torch.tensor(x['pesq_i'], dtype=torch.float32) for x in outputs]).mean()
        stoi_i_mean = torch.stack([torch.tensor(x['stoi_i'], dtype=torch.float32) for x in outputs]).mean()
    
        self.log('val_loss', val_loss_mean)
        self.log('val_pesq', pesq_i_mean)
        self.log('val_stoi', stoi_i_mean)

    def test_epoch_end(self, outputs):
        pesq_i_mean = torch.stack([torch.tensor(x['pesq_i'], dtype=torch.float32) for x in outputs]).mean()
        stoi_i_mean = torch.stack([torch.tensor(x['stoi_i'], dtype=torch.float32) for x in outputs]).mean()
    
        self.log('test_pesq', pesq_i_mean)
        self.log('test_stoi', stoi_i_mean)
        
    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(0.9, self.hparams.beta2))
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), self.hparams.learning_rate)
        else:
            raise ValueError(f"optimizer {self.hparams.optimizer} not implemented")

        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=len(self.train_dataloader()),
                epochs=self.hparams.max_epochs, anneal_strategy='linear')
            scheduler = {"scheduler": scheduler, "interval" : "step" }
            return [optimizer], [scheduler]
        else:
            return [optimizer]        
    
    def get_transform(self):        
        # data augment
        augments = list()
        if self.hparams.remix:
            augments.append(Remix())
        if self.hparams.shift:
            augments.append(Shift(self.hparams.shift, self.hparams.shift_same))
        if self.hparams.bandmask:
            augments.append(BandMask(self.hparams.bandmask, sample_rate=self.hparams.sample_rate))
        # if self.hparams.revecho:
        #     augments.append(RevEcho(self.hparams.revecho))

        augment_transform = torch.nn.Sequential(*augments)

        return augment_transform

    def prepare_data(self):
        segment_sec = int(self.hparams.segment_sec * self.hparams.sample_rate)
        hop_sec = int(self.hparams.hop_sec * self.hparams.sample_rate)

        length = self.valid_length(segment_sec)
        kwargs = {"sample_rate": self.hparams.sample_rate}
        
        # augmentation transformation for training
        self.augment_transform = self.get_transform()

        # Building datasets and loaders
        self.training_dataset = NoisyCleanSet(self.hparams.train, length=length, stride=hop_sec, pad=self.hparams.pad, **kwargs) # on découpe pour le training
        self.val_dataset = NoisyCleanSet(self.hparams.valid, **kwargs)
        self.test_dataset = NoisyCleanSet(self.hparams.test, **kwargs)

    def train_dataloader(self):        
        kwargs = {'num_workers': self.hparams.n_worker, 'pin_memory': True}
        training_dataloader = DataLoader(self.training_dataset, shuffle=True,
                                         batch_size=self.hparams.batch_size,
                                         **kwargs)
        return training_dataloader

    def val_dataloader(self):        
        kwargs = {'num_workers': self.hparams.n_worker, 'pin_memory': True}
        val_dataloader = DataLoader(self.val_dataset, shuffle=False,
                                    batch_size=1,
                                    **kwargs)
        return val_dataloader

    def test_dataloader(self):        
        kwargs = {'num_workers': self.hparams.n_worker, 'pin_memory': True}
        test_dataloader = DataLoader(self.test_dataset, shuffle=False,
                                    batch_size=1,
                                    **kwargs)
        return test_dataloader
