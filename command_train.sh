export CUDA_VISIBLE_DEVICES=0

export EXP_ID="music_vanilla_less_consistency_short"
python train.py \
	   --experiment_id $EXP_ID \
	   --dataset MUSIC \
   	   --num_epochs 100 \
	   --batchSize 16 \
	   --num_batch 30000 \
  	   --softmax_constraint False \
   	   --consistency_loss_weight 20 \
   	   --mask_loss_type L1 \
	   --num_disc_updates 1 \
	   --nThreads 16 \
	   --visual_pool conv1x1 \
	   --classifier_pool maxpool \
   	   --unet_num_layers 7 \
   	   --number_of_classes 15 \
	   --logscale_freq True \
	   --lr_visual 0.0001 \
	   --lr_unet 0.001 \
	   --lr_classifier 0.001 \
	   --optimizer adam \
	   --beta1 0.9 \
	   --weight_decay 0.0001 \
   	   --tensorboard True \
	   --audio_window 6946 \
	   --stft_frame 512 \
	   --stft_hop 160 \
	   --preload False |& tee logs/$EXP_ID.txt

