grep "iteration" slurm-26191.out

grep iteration slurm-6553.out | wc -l
grep iteration slurm-6553.out | awk '{sum+=$4} END {print sum}'

./profile.sh slurm-26191.out | awk '{print $3}'

./profile.sh slurm-26191.out &> tmp && cat tmp | awk '{sum+=$3}END { print sum}'

#check profile log cover 65%
cat encoder_f.log | awk '{sum+=$3}END { print sum}'

sed  -i "s/self.count == 20/self.count == 200/g" ./*.lua
