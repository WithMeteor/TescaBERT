# run noise inject

for i in $(seq 1 10)
do
   noise=$(echo "scale=1; $i/10" | bc)
   path="../config/TescaBERT_bank.json"
   python ../src/run.py --config $path --inject_train --noise_rate $noise
done
