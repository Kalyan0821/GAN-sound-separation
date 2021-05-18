python train.py \
	   --dataset MUSIC \
	   --experiment_id music_vanilla \
	   --batchSize 16 \
	   --nThreads 8 \
	   --tensorboard True \
	   --num_epochs 5 \
	   --num_batch 30000 \
	   --visual_pool conv1x1 \
	   --classifier_pool maxpool \
   	   --unet_num_layers 7 \
   	   --number_of_classes 15 \
   	   --consistency_loss_weight 0.05 \
   	   --mask_loss_type L1 \
	   --logscale_freq True \
	   --lr_visual 0.0001 \
	   --lr_unet 0.001 \
	   --lr_classifier 0.001 \
	   --optimizer adam \
	   --beta1 0.5 \
	   --weight_decay 0.0001 \
	   --num_disc_updates 1 |& tee -a log.txt





