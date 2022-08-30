# run TescaBertLda

# run model on bank
# python ../src/run.py  --config ../config/TescaBERTLda_bank.json --topic_num 5  --lda_model

# run model on asap
# python ../src/run.py  --config ../config/TescaBERTLda_asap.json --topic_num 5  --lda_model

# run model on sst5
# python ../src/run.py  --config ../config/TescaBERTLda_sst5.json --topic_num 5  --lda_model

# run model on food
# python ../src/run.py  --config ../config/TescaBERTLda_food.json --topic_num 5  --lda_model

# run model under different topic num
for i in $(seq 3 6)
do
   path="../config/TescaBERTLda_sst5.json"  # TescaBERTLda_sst5 or TescaBERTLda_food
   python ../src/run.py --config $path --topic_num $i --lda_model
done