export CUDA_VISIBLE_DEVICES=3

python train.py \
	   --experiment_id music_vanilla_100 \
	   --dataset MUSIC \
   	   --num_epochs 100 \
	   --batchSize 16 \
	   --num_batch 30000 \
   	   --consistency_loss_weight 20 \
   	   --softmax_constraint False \
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
	   --beta1 0.5 \
	   --weight_decay 0.0001 \
   	   --tensorboard True |& tee logs/music_vanilla_100.txt





