import argparse
import os
import torch
import torch.nn as nn
from util import get_config, mu_law_decode
import datetime
import shutil
from torch.utils.data import DataLoader
from dataset import VCTKDataset
from vctk import VCTK
from models import Encoder, Decoder, SpeechEncoder, VQVAE, VectorQuantizer,SpeechVQVAE
from wavenet_vocoder.wavenet import WaveNet
from tqdm import tqdm
import librosa
import numpy as np

def train_VCTK(opt):
    params = get_config(opt.config)

    save_path = os.path.join(params['save_path'], datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(save_path, exist_ok=True)
    shutil.copy('models.py', os.path.join(save_path, 'models.py'))
    shutil.copy('train.py', os.path.join(save_path, 'train.py'))
    shutil.copy(opt.config, os.path.join(save_path, os.path.basename(opt.config)))

    cuda = torch.cuda.is_available()
    gpu_ids = [i for i in range(torch.cuda.device_count())]

    TensorType = torch.cuda.FloatTensor if cuda else torch.Tensor

    vctk = VCTK(params['data_root'], ratio=params['train_val_split'])

    train_dataset = VCTKDataset(vctk.audios_train, vctk.speaker_dic, params)
    val_dataset = VCTKDataset(vctk.audios_val, vctk.speaker_dic, params)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'] * len(gpu_ids), shuffle=True,
                              num_workers=params['num_workers'], pin_memory=cuda)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=params['num_workers'], pin_memory=cuda)

    # models
    encoder = SpeechEncoder(params['d'])
    decoder = WaveNet(params['quantize'],
                      params['n_layers'],
                      params['n_loop'],
                      params['residual_channels'],
                      params['gate_channels'],
                      params['skip_out_channels'],
                      params['filter_size'],
                      cin_channels=params['local_condition_dim'],
                      gin_channels=params['global_condition_dim'],
                      n_speakers=len(train_dataset.speaker_dic),
                      upsample_conditional_features=True,
                      upsample_scales=[2, 2, 2, 2, 2, 2]) # 64 downsamples

    vq = VectorQuantizer(params['k'], params['d'], params['beta'], params['decay'], TensorType)

    model = SpeechVQVAE(encoder, decoder, vq)

    if params['checkpoint'] != None:
        checkpoint = torch.load(params['checkpoint'])

        params['start_epoch'] = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        vq.load_state_dict(checkpoint['vq'])

    if cuda:
        model = nn.DataParallel(model.cuda(), device_ids=gpu_ids)

    parameters = list(model.parameters())
    opt = torch.optim.Adam([p for p in parameters if p.requires_grad], lr=params['lr'])

    for epoch in range(params['start_epoch'], params['num_epochs']):
        train_bar = tqdm(train_loader)
        for data in train_bar:
            x_enc, x_dec, speaker_id, quantized = data
            if cuda:
                x_enc, x_dec, speaker_id, quantized = x_enc.cuda(), x_dec.cuda(), speaker_id.cuda(), quantized.cuda()

            opt.zero_grad()
            loss, _ = model(x_enc, x_dec, speaker_id, quantized)
            loss.mean().backward()
            opt.step()

            train_bar.set_description('Epoch {}: loss {:.4f}'.format(epoch + 1, loss.mean().item()))

        model.eval()
        data_val = next(iter(val_loader))
        x_enc_val, x_dec_val, speaker_id_val, quantized_val = data_val
        if cuda:
            x_enc_val, x_dec_val, speaker_id_val, quantized_val = x_enc_val.cuda(), x_dec_val.cuda(), speaker_id_val.cuda(), quantized_val.cuda()
        loss_val, out = model(x_enc_val, x_dec_val, speaker_id_val, quantized_val)

        output = out.argmax(dim=1).detach().cpu().numpy().squeeze()
        input_mu = x_dec_val.argmax(dim=1).detach().cpu().numpy().squeeze()
        input = x_enc_val.detach().cpu().numpy().squeeze()

        output = mu_law_decode(output)
        input_mu = mu_law_decode(input_mu)

        librosa.output.write_wav(os.path.join(save_path, '{}_output.wav'.format(epoch)), output, params['sr'])
        librosa.output.write_wav(os.path.join(save_path, '{}_input_mu.wav'.format(epoch)), input_mu, params['sr'])
        librosa.output.write_wav(os.path.join(save_path, '{}_input.wav'.format(epoch)), input, params['sr'])

        model.train()

        torch.save({'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'vq': vq.state_dict()
                    }, os.path.join(save_path, '{}_checkpoint.pth'.format(epoch)))

def train_CIFAR10(opt):

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    params = get_config(opt.config)

    save_path = os.path.join(params['save_path'], datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(save_path, exist_ok=True)
    shutil.copy('models.py', os.path.join(save_path, 'models.py'))
    shutil.copy('train.py', os.path.join(save_path, 'train.py'))
    shutil.copy(opt.config, os.path.join(save_path, os.path.basename(opt.config)))

    cuda = torch.cuda.is_available()
    gpu_ids = [i for i in range(torch.cuda.device_count())]

    TensorType = torch.cuda.FloatTensor if cuda else torch.Tensor

    data_path = os.path.join(params['data_root'], 'cifar10')

    os.makedirs(data_path, exist_ok=True)

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))

    val_dataset = datasets.CIFAR10(root=data_path, train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'] * len(gpu_ids), shuffle=True,
                              num_workers=params['num_workers'], pin_memory=cuda)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=params['num_workers'], pin_memory=cuda)

    data_variance = np.var(train_dataset.train_data / 255.0)

    encoder = Encoder(params['dim'], params['residual_channels'],params['n_layers'],params['d'])
    decoder = Decoder(params['dim'], params['residual_channels'],params['n_layers'],params['d'])

    vq = VectorQuantizer(params['k'], params['d'], params['beta'], params['decay'], TensorType)

    if params['checkpoint'] != None:
        checkpoint = torch.load(params['checkpoint'])

        params['start_epoch'] = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        vq.load_state_dict(checkpoint['vq'])


    model = VQVAE(encoder, decoder, vq)

    if cuda:
        model = nn.DataParallel(model.cuda(), device_ids=gpu_ids)

    parameters = list(model.parameters())
    opt = torch.optim.Adam([p for p in parameters if p.requires_grad], lr=params['lr'])

    for epoch in range(params['start_epoch'], params['num_epochs']):
        train_bar = tqdm(train_loader)
        for data, _ in train_bar:
            if cuda:
                data = data.cuda()
            opt.zero_grad()

            vq_loss, data_recon, _ = model(data)
            recon_error = torch.mean((data_recon - data) ** 2) / data_variance
            loss = recon_error + vq_loss.mean()
            loss.backward()
            opt.step()

            train_bar.set_description('Epoch {}: loss {:.4f}'.format(epoch + 1, loss.mean().item()))

        model.eval()
        data_val = next(iter(val_loader))
        data_val,_ = data_val

        if cuda:
            data_val = data_val.cuda()
        _, data_recon_val, _ = model(data_val)

        plt.imsave(os.path.join(save_path, 'latest_val_recon.png'),
                   (make_grid(data_recon_val.cpu().data) + 0.5).numpy().transpose(1, 2, 0))
        plt.imsave(os.path.join(save_path, 'latest_val_orig.png'),
                   (make_grid(data_val.cpu().data) + 0.5).numpy().transpose(1, 2, 0))

        model.train()

        torch.save({'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'vq': vq.state_dict(),
                    }, os.path.join(save_path, '{}_checkpoint.pth'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A pytorch implementation of VQ-VAE')
    parser.add_argument('--config', default='configs/speech_vctk.yaml', type=str, help='train batch size')

    opt = parser.parse_args()

    if 'vctk' in opt.config:
        train_VCTK(opt)
    elif 'cifar10' in opt.config:
        train_CIFAR10(opt)
    else:
        print('invalid config file')











