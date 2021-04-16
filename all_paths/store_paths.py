import os
import h5py

dataset = "MUSIC"  # set to "MUSIC"/"FAIR-Play"

if not os.path.isdir(dataset):
    os.makedirs(dataset)

if dataset == "MUSIC":
    top_detections_root = "/datasets/Kranthi/KALYAN/DATASETS/MUSIC_dataset/TOP_detection_results"
    train_npys = []
    train_solo_npys = []
    val_npys = []
    test_npys = []

    for ins in os.listdir(top_detections_root):
        ins_path = os.path.join(top_detections_root, ins)
        for vid in os.listdir(ins_path):
            vid_path = os.path.join(ins_path, vid)
            for clip_detection in os.listdir(vid_path):
                clip_detection_path = os.path.join(vid_path, clip_detection)

                if int(vid.split('_')[0]) == 1:  # set aside 1st video for validation
                    val_npys.append(clip_detection_path)
                elif int(vid.split('_')[0]) == 2:  # set aside 2nd video for testing
                    test_npys.append(clip_detection_path)
                else:
                    train_npys.append(clip_detection_path)
                    if '-' not in ins:  # only 1 instrument
                        train_solo_npys.append(clip_detection_path)

    print(len(train_npys), len(val_npys), len(test_npys), len(train_solo_npys))

    with open(os.path.join(dataset, "train.txt"), 'w') as f:
        for npy in train_npys:
            f.write(f"{npy}\n")

    with open(os.path.join(dataset, "val.txt"), 'w') as f:
        for npy in val_npys:
            f.write(f"{npy}\n")

    with open(os.path.join(dataset, "test.txt"), 'w') as f:
        for npy in test_npys:
            f.write(f"{npy}\n")

    with open(os.path.join(dataset, "train_solos.txt"), 'w') as f:
        for npy in train_solo_npys:
            f.write(f"{npy}\n")


elif dataset == "FAIR-Play":
    # top_detections_root = "/datasets/Kranthi/KALYAN/DATASETS/FAIR-Play_instruments/TOP_detection_results"
    pass







