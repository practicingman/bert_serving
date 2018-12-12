export BERT_BASE_DIR='chinese_L-12_H-768_A-12'
python3 test.py \
		--max_seq_length=256 \
    --predict_file=chinese_model/predict.tf_record \
		--export_dir=exported/1544585389
