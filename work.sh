python main.py \
    --task binary \
    --net ResNet50 \
    --output_dir ./output/Select/ResNet50-bin \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net ResNet50 \
    --output_dir ./output/Select/ResNet50-sub \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNet50 \
    --output_dir ./output/Select/ResNet50-bin-da \
    --batch_size 32 \
    --lr 1e-4 \
    --da \
    --resume

python main.py \
    --task subtype \
    --net ResNet50 \
    --output_dir ./output/Select/ResNet50-sub-da \
    --batch_size 32 \
    --lr 1e-4 \
    --da \
    --resume

python main.py \
    --task binary \
    --net VGG11 \
    --output_dir ./output/Select/VGG11-bin \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net VGG11 \
    --output_dir ./output/Select/VGG11-sub \
    --batch_size 32 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet121 \
    --output_dir ./output/Select/DenseNet121-bin \
    --batch_size 16 \
    --lr 1e-5 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet121 \
    --output_dir ./output/Select/DenseNet121-sub \
    --batch_size 16 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNet152 \
    --output_dir ./output/Select/ResNet152-bin \
    --batch_size 16 \
    --lr 1e-5 \
    --resume

python main.py \
    --task subtype \
    --net ResNet152 \
    --output_dir ./output/Select/ResNet152-sub \
    --batch_size 16 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net DenseNet201 \
    --output_dir ./output/Select/DenseNet201-bin \
    --batch_size 16 \
    --lr 1e-5 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet201 \
    --output_dir ./output/Select/DenseNet201-sub \
    --batch_size 16 \
    --lr 1e-4 \
    --resume

python main.py \
    --task subtype \
    --net DenseNet201 \
    --output_dir ./output/Select/DenseNet201-sub-da \
    --batch_size 16 \
    --lr 1e-4 \
    --da \
    --resume

python main.py \
    --task binary \
    --net VGG19_bn \
    --output_dir ./output/Select/VGG19_bn-bin \
    --batch_size 16 \
    --lr 1e-5 \
    --resume

python main.py \
    --task subtype \
    --net VGG19_bn \
    --output_dir ./output/Select/VGG19_bn-sub \
    --batch_size 16 \
    --lr 1e-4 \
    --resume

python main.py \
    --task binary \
    --net ResNeXt_101_32x8d \
    --output_dir ./output/Select/ResNeXt_101_32x8d-bin \
    --batch_size 8 \
    --lr 1e-5 \
    --resume

python main.py \
    --task subtype \
    --net ResNeXt_101_32x8d \
    --output_dir ./output/Select/ResNeXt_101_32x8d-sub \
    --batch_size 8 \
    --lr 1e-4 \
    --resume