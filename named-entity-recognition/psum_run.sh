# DATA_DIR=/data/joan/datasets/NER
# ENTITY=sdNER/

DATA_DIR=../datasets/NER
ENTITY=sdNER

CUDA_VISIBLE_DEVICES=1

python -c "import torch; print('using pytorch', torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

python psum_run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY} \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.2 \
    --output_dir output/${ENTITY} \
    --max_seq_length 256 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --save_steps 5000 \
    --seed 1 \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir \
    --overwrite_cache


