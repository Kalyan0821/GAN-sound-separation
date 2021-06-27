export CUDA_VISIBLE_DEVICES=0

python test.py \
	   --experiment_id music_vanilla_less_consistency \
	   --epoch 8 \
  	   --softmax_constraint False \
	   --dataset MUSIC \
	   --visual_pool conv1x1 \
	   --classifier_pool maxpool \
   	   --unet_num_layers 7 \
   	   --number_of_classes 15 \
	   --logscale_freq True



