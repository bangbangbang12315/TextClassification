DEVICES="1"

TRAIN_PATH=../data/train.txt
DEV_PATH=../data/dev.txt
SRC_VOCAB=../data/vocab

# Start training
python run.py \
        --device $DEVICES \
        --train_path $TRAIN_PATH \
        --dev_path $DEV_PATH \
        --src_vocab_file $SRC_VOCAB \
        --bidirectional \
        --random_seed 2808 \
        --model_dir './experiment_trans' \
        --best_model_dir './experiment_trans/best' \
        --model_name transformer\
        # --resume
