##VQ-VAE
A pytorch implementation of VQ-VAE based on the paper [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

##Requirements
* pytorch >=0.4
* librosa
* yaml
* tqdm

##Dataset

Dataset will be automatically downloaded
* VCTK
* CIFAR10

##Training for speech (Multi-GPUs support)
```
python train.py --config confgs/speech_vctk.yaml
```
##Generating
```
python generate.py --config out/image_cifar10.yml --output_path gen_output --speaker 10 --num_samples 10
```

##Training for image (Multi-GPUs support)
```
python train.py --config confgs/image_cifar10.yaml  
```

##To do
- [ ] Multi-GPUs for generating
- [ ] Fast generating algorithm
- [ ] Upload results 