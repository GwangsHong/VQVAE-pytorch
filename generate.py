import torch
from models import Encoder, VectorQuantizer, ConditionEmbedding, WaveNet
import numpy as np
from util import mu_law_decode ,get_config
import librosa
import pathlib
from vctk import VCTK
from dataset import VCTKDataset
from torch.utils.data import DataLoader
import argparse

from tqdm import tqdm
import os
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Test VQ-VAE')
    parser.add_argument('--config', default='configs/speech_vctk.yaml', type=str, help='train batch size')
    parser.add_argument('--output_path', default = 'gen_output', type=str, help='output file')
    parser.add_argument('--speaker', default=0 ,help='name of speaker or id of speaker')
    parser.add_argument('--num_samples', default=10, type=int, help= 'number of generated samples')
    opt = parser.parse_args()
    params = get_config(opt.config)

    os.makedirs(opt.output_path,exist_ok=True)

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
    encoder = Encoder(params['d'])
    decoder = WaveNet(params['n_loop'],
                      params['n_layer'],
                      params['filter_size'],
                      params['input_dim'],
                      params['residual_channels'],
                      params['dilated_channels'],
                      params['skip_channels'],
                      params['quantize'],
                      params['local_condition_dim'] + params['global_condition_dim'])
    condition_emb = ConditionEmbedding(len(test_dataset.speaker_dic), params['global_condition_dim'],
                                       params['local_condition_dim'])
    vq = VectorQuantizer(params['k'], params['d'], params['beta'], TensorType)
    #
    checkpoint = torch.load(params['checkpoint'])

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    condition_emb.load_state_dict(checkpoint['condition_emb'])
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
        condition_emb.cuda().eval()
        vq.cuda().eval()

    test_bar = tqdm(test_loader)
    for idx, (x_enc,_,_,_) in enumerate(test_bar):
        if cuda:
            x_enc, global_condition = x_enc.cuda(), global_condition.cuda()

        x_dec = torch.zeros(params['batch_size'], params['input_dim'], 1, 1).type(TensorType)
        z = encoder(x_enc)
        _, local_condition, _, _ = vq(z)
        condition = condition_emb(local_condition, global_condition)
        decoder.init(params['batch_size'], TensorType)

        output = np.zeros(condition.shape[2])
        for i in range(len(output) - 1):

            out = decoder.generate(x_dec,  condition[:, :, i:i + 1])
            value = np.random.choice(params['quantize'], p=out.softmax(dim=1)[0, :, 0, 0].detach().cpu().numpy())
            x_dec.zero_()
            x_dec[:,value,:,:] = 1
            output[i] = value

        wave = mu_law_decode(y=output)
        librosa.output.write_wav(os.path.join(opt.output_path, '{}_input.wav'.format(idx)), x_enc.detach().cpu().numpy().squeeze(), params['sr'])
        librosa.output.write_wav(os.path.join(opt.output_path,'{}_output.wav'.format(idx)), wave, params['sr'])
