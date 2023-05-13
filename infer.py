import sys

sys.path.append('./VAE_Tacotron2_korean')
sys.path.append('./HiFi_GAN')

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
import matplotlib.pylab as plt
import src.models.utils as models
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
from src.models.tacotron2.text import text_to_sequence
from train import parse_args
from src.utils.common.utils import load_wav_to_torch
from src.utils.common.layers import TacotronSTFT
from matplotlib.pyplot import imsave
from PIL import Image

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(checkpoint_file, input_mels_dir, output_dir, iter_num):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(input_mels_dir)

    os.makedirs(output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            x = np.load(os.path.join(input_mels_dir, filname))
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(output_dir, os.path.splitext(filname)[0] + f'_generated_e2e_{iter_num}.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def load_and_setup_model(model_name, parser, checkpoint, fp16_run, cpu_run, forward_is_infer=False):
    model_parser = models.model_parser(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, cpu_run=cpu_run,
                             forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        if cpu_run:
            state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            state_dict = torch.load(checkpoint)['state_dict']

        model.load_state_dict(state_dict)

    model.eval()

    if fp16_run:
        model.half()

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, cpu_run=False):
    d = []
    for i, text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['korean_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths


def load_mel(path):
    stft = TacotronSTFT()
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != 16000:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / 32768.0  # hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def save_mel_for_hifigan(mel):
    # print(mel.shape)
    # print(type(mel))
    mel = mel.float().data.cpu().numpy()[0]
    np.save('/data/VAE_taco/vae-taco-hifi-gan/HiFi_GAN/test_mel_files/result.npy', mel)


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def read_text_file(dir, data):
    file = open(dir, "r", encoding="utf-8")
    while True:
        line = file.readline()
        if not line:
            file.close()
            break
        data.append(line)
    return data

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_data(data, num, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')
    path = 'results' + str(num) + '.png'
    fig.savefig(path)

def plot_alignment_to_numpy(alignment, info=None):
    # Figure confinguration
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()


    data = save_figure_to_numpy(fig)
    plt.close()
    return data

# 유사도 검사 함수
def check_sentence_similarity(main_sentence, sentences):
    best_sentence = ""
    best_similarity = 0

    #for sentence in sentences:





#



# text를 입력으로 받고 filelist를 참조해서 입력으로 받은 문장과 가장 유사한 데이터를 참조




# text를 입력으로 받아서 inference를 통해 wav 파일을 생성 하는 함수 textToWav





emotion_dict = {
        '무감정': 0,
        '당황':1,
        '기쁨':2,
        '불안':3,
        '분노':4,
        '슬픔':5,
        '상처':6}

emotion_ids = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
speaker_ids = {'06':0,'12':1,"19":2,'21':3,'28':4,'30':5,'31':6,'36':7,'42':8,'43':9,'48':10}

# 변수 선언
checkpoint_path_vaetaco = "/data/VAE_taco/vae-taco-hifi-gan/VAE_Tacotron2_korean/output/checkpoint_Tacotron2_JLT_FINAL_30.pt" # vaetaco의 cp
checkpoint_path_hifigan = '/data/VAE_taco/vae-taco-hifi-gan/HiFi_GAN/chk_pt/Universal/g_02500000' # hifigan의 cp
input_mels_dir = '/data/VAE_taco/vae-taco-hifi-gan/HiFi_GAN/test_mel_files' # 접두사
output_dir = '/data/infer_result' # 출력 wav의 결과 디렉토리
input_text_file_path = "/data/VAE_taco/vae-taco-hifi-gan/VAE_Tacotron2_korean/infer_text.txt" #  테스트할 텍스트가 포함된 문장, |로 구분되어 text, emotion, speaker로 구성된다.
config_file = '/data/VAE_taco/vae-taco-hifi-gan/HiFi_GAN/chk_pt/Universal/config.json' # config file
ref_mels_path = '/data/VAE_taco/vae-taco-hifi-gan/data_sample/*.wav' # vae 와 구조를 맞춰주기 위해서 sample reference audio가 필요

is_fp16 = True
is_cpu = False

parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Inference')
parser = parse_args(parser)
args, _ = parser.parse_known_args()
tacotron2 = load_and_setup_model('Tacotron2', parser, checkpoint_path_vaetaco,
                                 is_fp16, is_cpu,
                                 forward_is_infer=True)  # forward is infer를 해줌으로써 tacotron model의 infer로 간다.

jitted_tacotron2 = torch.jit.script(tacotron2)

ref_mels = glob.glob(ref_mels_path) # 해당 경로의 디렉토리에 존재하는 확장자를 가진 파일들을 리스트로 반환
texts = []
texts = read_text_file(input_text_file_path, texts)
# input_text_file_path의 경로에 존재하는 txt 파일을 읽고 문장을 토큰화하여 texts에 저장

for i, text in enumerate(texts):
    ref_mels[i].split('/')[1]
    text = text.split('|')
    emotion_id = emotion_dict[text[1]] # 첫번째는 emotion
    speaker_id = ref_mels[i].split('/')[-1].split('-')[2] # '/'으로 split 하는 이유는 이 문자가 디렉토리를 구분할때 사용하기 때문
    # split된 결과의 마지막은 파일명을 의미함, split('-')으로 분리했을때 성우 아이디가 대입되어야함
    # ex) N0001-03-01-00.wav => N0001 / 03 / 01(ID) / 00
    emotion_id = to_gpu(torch.IntTensor([emotion_ids[str(emotion_id)]])).long()
    speaker_id = to_gpu(torch.IntTensor([speaker_ids[speaker_id]])).long()

    ref_mel = load_mel(ref_mels[i])
    measurements = {}
    sequences_padded, input_lengths = prepare_input_sequence([text[2]], is_cpu) # 두번째는 text # TODO: EDITED text[2] -> text[0]

    with torch.no_grad():
        mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths, ref_mel, emotion_id,
                                                        speaker_id) # TODO: 여기서 문제 발생!!
    # print("#alignments.shape#")
    # print(alignments.shape)
    # print("#alignments#")
    # print(alignments)
    #
    #
    # print("#size#")
    # print(alignments.size(0))
    # print("#Transpose#")
    #
    # data = plot_alignment_to_numpy(alignments.data.cpu().numpy().T)
    # print(data.shape)
    #
    # # save png
    # root = "/data/VAE_taco/vae-taco-hifi-gan/for_alignment"
    # imgpath = root + "/" + "alignment_" + str(i) + ".png"
    # imsave(imgpath, data)

    plot_data((mel.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T), i)

    save_mel_for_hifigan(mel)

    # VOCODER
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(checkpoint_path_hifigan, input_mels_dir, output_dir, i)