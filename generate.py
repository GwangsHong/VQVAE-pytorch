import torch
from models import SpeechEncoder, VectorQuantizer
from wavenet_vocoder.wavenet import WaveNet
import numpy as np
from util import mu_law_decode ,get_config
import librosa
import pathlib
from vctk import VCTK
from dataset import VCTKDataset
from torch.utils.data import DataLoader
import argparse
from util import Timer
from tqdm import tqdm
import os
def wavegen(model,length=None, c=None, g=None, initial_input=None,
            fast=False, tqdm=tqdm):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (Tensor): Conditional features, of shape B x C x T
        g (Tensor): Speaker ID
        initial_value (Tensor) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """

    model.eval()
    if fast:
        model.make_generation_fast_()

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True)

    return y_hat

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Test VQ-VAE')
    parser.add_argument('--config', default='configs/speech_vctk_test.yaml', type=str, help='train batch size')
    parser.add_argument('--output_path', default = 'gen_output', type=str, help='output file')
    parser.add_argument('--speaker', default=0 ,help='name of speaker or id of speaker')
    parser.add_argument('--num_samples', default=9, type=int, help= 'number of generated samples')
    opt = parser.parse_args()
    params = get_config(opt.config)

    output_path = os.path.join(opt.output_path,os.path.basename(params['checkpoint'])[:-4])
    os.makedirs(output_path,exist_ok=True)

    cuda = torch.cuda.is_available()
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    TensorType = torch.cuda.FloatTensor if cuda else torch.Tensor

    n_speaker = len([speaker for speaker in pathlib.Path(params['data_root']).glob('*/*/*/wav48/*/')])

    assert params['checkpoint'] != None
    params['length'] = None
    params['batch_size'] = 1

    vctk = VCTK(params['data_root'])

    test_dataset = VCTKDataset(vctk.audios[:opt.num_samples], vctk.speaker_dic, params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], num_workers=params['num_workers'], pin_memory=cuda)

    #models
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
                      n_speakers=len(test_dataset.speaker_dic),
                      upsample_conditional_features=True,
                      upsample_scales=[2, 2, 2, 2, 2, 2])  # 64 downsamples

    vq = VectorQuantizer(params['k'], params['d'], params['beta'], TensorType)
    #
    checkpoint = torch.load(params['checkpoint'])

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    vq.load_state_dict(checkpoint['vq'])

    if type(opt.speaker) == int:
        assert (opt.speaker in test_dataset.speaker_dic.values()), 'invalid speaker'
    else:
        assert (opt.speaker  in test_dataset.speaker_dic.keys()),'invalid speaker'
        opt.speaker = test_dataset.speaker_dic[opt.speaker]

    speaker_id = np.array([opt.speaker])
    global_condition = torch.from_numpy(speaker_id).long()

    if cuda:
        encoder.cuda().eval()
        decoder.cuda().eval()
        vq.cuda().eval()

    test_bar = tqdm(test_loader)
    for idx, (x_enc,_,_,t) in enumerate(test_bar):
        if cuda:
            x_enc, global_condition = x_enc.cuda(), global_condition.cuda()

        x_dec = torch.zeros(params['batch_size'], params['input_dim'], 1, 1).type(TensorType)
        z = encoder(x_enc)
        _, local_condition, _, _ = vq(z)
        local_condition = local_condition.squeeze(-1)

        y_hat = wavegen(decoder,length = local_condition.size(-1) * 64,c = local_condition,g = global_condition, initial_input=x_dec)
        y = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        waveform = mu_law_decode(y)

        librosa.output.write_wav(os.path.join(output_path, '{}_input.wav'.format(idx)), x_enc.detach().cpu().numpy().squeeze(), params['sr'])
        librosa.output.write_wav(os.path.join(output_path,'{}_output.wav'.format(idx)), waveform, params['sr'])
