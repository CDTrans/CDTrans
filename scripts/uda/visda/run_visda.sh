model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='uda_vit_base_patch16_224_TransReID'
    gpus="('0,1')"
else
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    gpus="('0')"
fi
python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
OUTPUT_DIR '../logs/uda/'$model'/visda/' \
MODEL.PRETRAIN_PATH '../logs/pretrain/'$model'/visda/transformer_10.pth' \
DATASETS.ROOT_TRAIN_DIR './data/visda/train/train_image_list.txt' \
DATASETS.ROOT_TRAIN_DIR2 './data/visda/validation/valid_image_list.txt' \
DATASETS.ROOT_TEST_DIR './data/visda/validation/valid_image_list.txt' \
DATASETS.NAMES "VisDA" DATASETS.NAMES2 "VisDA"  \
SOLVER.BASE_LR 0.00005 \
MODEL.Transformer_TYPE $model_type \

