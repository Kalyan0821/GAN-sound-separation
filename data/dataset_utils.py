import numpy as np
import random
from random import randrange
import librosa
import os
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch._six import container_abcs, string_classes, int_classes

def get_vid_path_MUSIC(npy_path):
    return '/'.join(npy_path.split('/')[:-1])

def get_audio_path_MUSIC(npy_path):
    return os.path.join('/'.join(npy_path.split('/')[:-4]), "clip_audios_11025", '/'.join(npy_path.split('/')[-3:]))[:-4] + ".wav"

def get_frames_path_MUSIC(npy_path):
    return os.path.join('/'.join(npy_path.split('/')[:-4]), "frames", '/'.join(npy_path.split('/')[-3:]))[:-4]

def get_ground_truth_labels_MUSIC(vid_path):
    return vid_path.split('/')[-2].split('-')


def sample_object_detections(detection_bbs):  # input: np.array from a single .npy clip-detection file
    class_index_clusters = {}  # {class_id, [frame_indices_list]} dict
    for i in range(detection_bbs.shape[0]):  # iterate over frames in the clip
        if int(detection_bbs[i, 1]) in class_index_clusters:
            class_index_clusters[int(detection_bbs[i, 1])].append(i)  # add each frame under one/more detected classes
        else:
            class_index_clusters[int(detection_bbs[i,1])] = [i]

    detection2return = np.array([])
    for cls in class_index_clusters.keys():  # iterate through all discovered classes
        sampledIndex = random.choice(class_index_clusters[cls])  # sample a single random frame for each class
        if detection2return.shape[0] == 0:
            detection2return = np.expand_dims(detection_bbs[sampledIndex, :], axis=0)
        else:
            detection2return = np.concatenate((detection2return, np.expand_dims(detection_bbs[sampledIndex,:], axis=0)), axis=0)

    return detection2return  # C x 7 array: return 1 random BB image (from entire clip) for each of the C discovered classes in the clip

def sample_audio(audio, window):
    # repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start:(audio_start+window)]
    return audio_sample

def augment_audio(audio):
    audio = audio * (random.random() + 0.5)  # 0.5 to 1.5
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase  # 1 x F x T
    else:
        return spectro_mag

def augment_image(image):
	if(random.random() < 0.5):
		image = image.transpose(Image.FLIP_LEFT_RIGHT)
	enhancer = ImageEnhance.Brightness(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	enhancer = ImageEnhance.Color(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	return image


def object_collate(examples_list):

    numpy_to_torch_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
    }

    elem_type = type(examples_list[0])

    if elem_type.__module__ == 'numpy':  # if each example is a numpy array/scalar
        elem = examples_list[0]
        if elem_type.__name__ == 'ndarray':
            return torch.cat([torch.from_numpy(b) for b in examples_list], dim=0)
        if elem.shape == ():
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_to_torch_map[elem.dtype.name](list(map(py_type, examples_list)))

    elif isinstance(examples_list[0], container_abcs.Mapping):  # if each example is a dict
        return {key: object_collate([d[key] for d in examples_list]) for key in examples_list[0]}  # return a dict (stack for each key)

    raise TypeError(f"Batch must contain tensors, numbers, dicts, or lists. Found {type(examples_list[0])}")


