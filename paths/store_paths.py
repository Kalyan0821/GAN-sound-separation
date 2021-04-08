import os
import h5py

dataset = "MUSIC"  # set to "MUSIC"/"FAIR-Play"

if not os.path.isdir(dataset):
	os.makedirs(dataset)

if dataset == "MUSIC":
    top_detections_root = "/datasets/Kranthi/KALYAN/DATASETS/MUSIC_dataset/TOP_detection_results"
    train_npys = []
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

    with h5py.File(os.path.join(dataset, "train.h5"), 'w') as f:
    	f.write(train_npys)












elif dataset == "FAIR-Play":
    # top_detections_root = "/datasets/Kranthi/KALYAN/DATASETS/FAIR-Play_instruments/TOP_detection_results"
    pass







