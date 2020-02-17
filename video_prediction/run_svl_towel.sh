PROBLEM=video_roboturk_stanford_dataset
MODEL=next_frame_sv2p
HPARAMS=custom_next_frame_sv2p

DATA_DIR={DATA_DIR}
TMP_DIR={TMP_DIR}
TRAIN_DIR={TARGET_DIR}/$PROBLEM/$MODEL-$HPARAMS
USR_DIR={TARGET_DIR}/roboturk_real_dataset/video_prediction

mkdir -p $TMP_DIR $TRAIN_DIR

t2t-trainer --data_dir=$DATA_DIR --problem=$PROBLEM --model=$MODEL --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --t2t_usr_dir=$USR_DIR

