
echo ${1} > tmp_cfg.txt

YOUR_CSV_PATH="/mnt/data/sora/meta/UCF/UCF-101/train/ApplyEyeMakeup/meta_clips_caption.csv"
cfg_file="configs/opensora-v1-1/train/stage1_tmp.py"
torchrun --standalone --nproc_per_node 8 scripts/fake_train.py \
    ${cfg_file} --data-path ${YOUR_CSV_PATH}

YOUR_CSV_PATH="/mnt/data/sora/meta/UCF/UCF-101/train/ApplyEyeMakeup/meta_clips_caption.csv"
cfg_file="configs/opensora-v1-1/train/stage1_tmp.py"
torchrun --standalone --nproc_per_node 8 scripts/fake_train.py \
    ${cfg_file} --data-path ${YOUR_CSV_PATH} --only-dit  > tmp.log

python3 tmp.py