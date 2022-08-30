# run loss rate

for i in $(seq 1 10)
do
   loss=$(echo "scale=1; $i/10" | bc)
   path="../config/TescaBERT_asap.json"
   python ../src/run.py --config $path --change_loss --loss_ratio $loss
done
