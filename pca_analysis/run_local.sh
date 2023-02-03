clipping_norm=1.0
dataset=$1
# for num_components in 10 25 50 100 125 250 500 1000
# do
for random_seed in 0 1 2 3 4 5
do
for sigma in 25.077 13.285 7.032 3.731 1.994 1.081 0.6 0.344
do
python main.py --dataset $dataset --clipping_norm $clipping_norm --random_seed $random_seed --setting local --sigma $sigma
python main.py --dataset $dataset --clipping_norm $clipping_norm --random_seed $random_seed --setting centralized --sigma $sigma
# done
done
done