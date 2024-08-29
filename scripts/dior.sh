SPLIT_ID=1 # <-- change split_id 1,2,3,4
SAVE_DIR=checkpoints/dior_dataset
SPLIT_NAME=split${SPLIT_ID}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth

# ------------------------------- Pre-train ----------------------------------
python3 main.py --num-gpus 1 --config-file configs/dior/ecfsod_det_r101_base${SPLIT_ID}.yaml \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                \
           OUTPUT_DIR ${SAVE_DIR}/${SPLIT_NAME}/pretrain/

# ----------------------------- Model Preparation ---------------------------------
python3 tools/model_surgery.py --dataset dior --method randinit                              \
    --src-path ${SAVE_DIR}/${SPLIT_NAME}/pretrain/model_final.pth                            \
    --save-dir ${SAVE_DIR}/${SPLIT_NAME}/prepare/
BASE_WEIGHT=${SAVE_DIR}/${SPLIT_NAME}/prepare/model_reset_surgery.pth

for n in 1 2 3 4 5
do
    seed=1
    for shot in 3 5 10 20
    do
        python3 tools/create_config.py --dataset dior --config_root configs/dior             \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}

        CONFIG_PATH=configs/dior/ecfsod_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/${SPLIT_NAME}/finetuning/number${n}/${shot}shot_seed${seed}
        
        python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
                   
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
    python3 tools/extract_results.py --res-dir ${SAVE_DIR}/${SPLIT_NAME}/finetuning/number${n} --shot-list 3 5 10 20  
done
