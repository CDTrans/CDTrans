model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='vit_base_patch16_224_TransReID'
    pretrain_model='deit_base_distilled_patch16_224-df68dfff.pth'
else
    model='vit_small'
    model_type='vit_small_patch16_224_TransReID'
    pretrain_model='deit_small_distilled_patch16_224-649709d9.pth'
fi
python train.py --config_file configs/pretrain.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES 'OfficeHome' \
OUTPUT_DIR '../logs/pretrain/vit_base/office-home/Real_World' \
DATASETS.ROOT_TRAIN_DIR './data/office31/Real_World.txt' \
DATASETS.ROOT_TEST_DIR './data/office31/Clipart.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH './data/pretrainModel/'$pretrain_model \
