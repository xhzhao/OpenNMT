grep "iteration" slurm-26191.out

./profile.sh slurm-26191.out | awk '{print $3}'

./profile.sh slurm-26191.out &> tmp && cat tmp | awk '{sum+=$3} { print sum}'
