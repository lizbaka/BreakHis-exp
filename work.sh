# ResNet50
python main.py \
    --task binary \
    --net ResNet50 \
    --output_dir ./output/Select/ResNet50 \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --output_dir ./output/Select/VGG11 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --output_dir ./output/Select/DenseNet121 \
    --resume

# ResNeXt_101_32x8d
python main.py \
    --task binary \
    --net ResNeXt_101_32x8d \
    --output_dir ./output/Select/ResNeXt_101_32x8d \
    --resume