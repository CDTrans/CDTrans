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
for target_dataset in 'infograph' 'painting' 'real' 'sketch' 'clipart'
do
    python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus\
    OUTPUT_DIR '../logs/uda/'$model'/domainnet/quickdraw2'$target_dataset  \
    MODEL.PRETRAIN_PATH '../logs/pretrain/'$model'/domainnet/Quickdraw/transformer_10.pth' \
    DATASETS.ROOT_TRAIN_DIR './data/domainnet/quickdraw.txt' \
    DATASETS.ROOT_TRAIN_DIR2 './data/domainnet/'$target_dataset'.txt' \
    DATASETS.ROOT_TEST_DIR './data/domainnet/'$target_dataset'.txt' \
    DATASETS.NAMES "DomainNet" DATASETS.NAMES2 "DomainNet" \
    MODEL.Transformer_TYPE $model_type \


done





