
def sample_object_detections(detection_bbs):  # input: np.array from a single .npy clip-detection file
    class_index_clusters = {}  # {class_id, [frame_indices_list]} dict
    for i in range(detection_bbs.shape[0]):  # iterate over frames in the clip
        if class_index_clusters.has_key(int(detection_bbs[i, 1])):
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
        return spectro_mag, spectro_phase
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


# Change these
def get_vid_name_MUSIC(npy_path):
    return os.path.basename(npy_path)[0:11]

def get_audio_root_MUSIC(npy_path):
    return os.path.join(os.path.dirname(os.path.dirname(npy_path)), 'audio_11025')

def get_clip_name_MUSIC(npy_path):
    return os.path.basename(npy_path)[0:-4]

def get_frame_root_MUSIC(npy_path):
    return os.path.join(os.path.dirname(os.path.dirname(npy_path)), 'frame')




