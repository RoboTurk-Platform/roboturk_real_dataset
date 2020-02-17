PROBLEM=video_bair_robot_pushing_with_actions_fixed

DATA_DIR={DATA_DIR}
TMP_DIR={TMP_DIR}
USR_DIR={TARGET_DIR}/roboturk_real_dataset/video_prediction

t2t-datagen --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR --problem=$PROBLEM --t2t_usr_dir=$USR_DIR
