import os

os.system("python -u DeepSpeech.py \
  --train_files '/deepspeech-test/ldc93s1/ldc93s1.csv' --train_batch_size 1 \
  --dev_files '/deepspeech-test/ldc93s1/ldc93s1.csv' --dev_batch_size 1 \
  --test_files '/deepspeech-test/ldc93s1/ldc93s1.csv' --test_batch_size 1 \
  --n_hidden 494 --epoch 75 --random_seed 4567 --default_stddev 0.046875 \
  --max_to_keep 1 --checkpoint_dir './tmp/ckpt' --checkpoint_secs 0 \
  --learning_rate 0.001 --dropout_rate 0.05  --export_dir './tmp/train' \
  --use_seq_length False --decoder_library_path './tmp/ds/libctc_decoder_with_kenlm.so'")

print("Done?")