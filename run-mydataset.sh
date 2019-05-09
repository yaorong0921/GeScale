python main.py --dataset SHGD \
               --modality IRD  \
               --arch mobilenetv2 \
               --num_segments 8 \
               --batch-size 24 \
               --epochs 60 \
               --img_feature_dim 64 \
               --lr 0.001 \
               --dropout 0.8 \
               --workers 8 \
               --pretrained './pretrained_models/mydata_GrayDepth_mobilenetv2_segment8_3f1c_best.pth.tar' \
               #--test \
               #--resume './models/mydata_GrayDepth_squeezenet1_1_for_tuples_93939.pth.tar' \
               #--pretrained './pretrained_models/jester_RGB_squeezenet1_1_segment8_3f1c_best.pth.tar'
               #--resume './model/mydata_GrayDepth_squeezenet1_1_segment8_3f1c_best.pth.tar' \
	       #--resume './model/mydata_GrayDepth_squeezenet1_1_segment8_3f1c_best.tar' \ --pretrained './pretrained_models/jester_RGB2_squeezenet1_1_segment8_3f1c_checkpoint.pth.tar' \
            # squeezenet1_1 shufflenet
