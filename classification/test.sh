DEVICES="1"

TEST_PATH=../data/test.txt
LOG_PATH=test.log
SRC_VOCAB=../data/vocab

# Start training
python run.py \
        --device $DEVICES \
        --test_path $TEST_PATH \
        --src_vocab_file $SRC_VOCAB \
        --random_seed 2808 \
        --log_file $LOG_PATH \
        --model_dir './experiment_trans' \
        --best_model_dir './experiment_trans/best' \
        --model_name transformer\
        --phase 'test' \
        --resume
