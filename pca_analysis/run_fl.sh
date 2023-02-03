clipping_norm=1.0
random_seed=0
dataset=$1
# for num_components in 10 25 50 100 125 250 500 1000
# do
for random_seed in 0 1 2 3 4 5
do
for b in 0 1 2 3 4 5 
do 
for skellam_mu in 382.3 106.1 29.4 8.19 2.32 0.671 0.2036 0.067
do
    python main.py --dataset $dataset --clipping_norm $clipping_norm --b $b --random_seed $random_seed --setting fl --skellam_mu $skellam_mu
done
done
done
# done

