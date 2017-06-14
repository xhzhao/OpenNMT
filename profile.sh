logName=$1
echo $logName

#get the iteration number
iter=$(grep "iteration = " $logName | wc -l)
echo $iter

grep iteration     $logName | awk '{sum+=$4}END {print "Iteration time    = " sum/'$iter'}'


grep Linear_F      $logName | awk '{sum+=$3}END {print "Linear_F    = " sum/'$iter'}'
grep Linear_B1     $logName | awk '{sum+=$3}END {print "Linear_B1   = " sum/'$iter'}'
grep Linear_B2     $logName | awk '{sum+=$3}END {print "Linear_B2   = " sum/'$iter'}'

grep Tanh_F        $logName | awk '{sum+=$3}END {print "Tanh_F      = " sum/'$iter'}'
grep Tanh_B        $logName | awk '{sum+=$3}END {print "Tanh_B      = " sum/'$iter'}'

grep Sigmoid_F     $logName | awk '{sum+=$3}END {print "Sigmoid_F   = " sum/'$iter'}'
grep Sigmoid_B     $logName | awk '{sum+=$3}END {print "Sigmoid_B   = " sum/'$iter'}'

grep MM_F          $logName | awk '{sum+=$3}END {print "MM_F        = " sum/'$iter'}'
grep MM_B          $logName | awk '{sum+=$3}END {print "MM_B        = " sum/'$iter'}'

grep CAddTable_F   $logName | awk '{sum+=$3}END {print "CAddTable_F = " sum/'$iter'}'
grep CAddTable_B   $logName | awk '{sum+=$3}END {print "CAddTable_B = " sum/'$iter'}'

grep CMulTable_F   $logName | awk '{sum+=$3}END {print "CMulTable_F = " sum/'$iter'}'
grep CMulTable_B   $logName | awk '{sum+=$3}END {print "CMulTable_B = " sum/'$iter'}'

grep JoinTable_F   $logName | awk '{sum+=$3}END {print "JoinTable_F = " sum/'$iter'}'
grep JoinTable_B   $logName | awk '{sum+=$3}END {print "JoinTable_B = " sum/'$iter'}'

grep SoftMax_F     $logName | awk '{sum+=$3}END {print "SoftMax_F   = " sum/'$iter'}'
grep SoftMax_B     $logName | awk '{sum+=$3}END {print "SoftMax_B   = " sum/'$iter'}'

grep LogsoftMax_F  $logName | awk '{sum+=$3}END {print "LogsoftMax_F = " sum/'$iter'}'
grep LogsoftMax_B  $logName | awk '{sum+=$3}END {print "LogsoftMax_B = " sum/'$iter'}'

grep ClassNLLCriterion_F  $logName | awk '{sum+=$3}END {print "ClassNLLCriterion_F = " sum/'$iter'}'
grep ClassNLLCriterion_B  $logName | awk '{sum+=$3}END {print "ClassNLLCriterion_B = " sum/'$iter'}'

grep Dropout_F  $logName | awk '{sum+=$3}END {print "Dropout_F = " sum/'$iter'}'
grep Dropout_B  $logName | awk '{sum+=$3}END {print "Dropout_B = " sum/'$iter'}'

grep SplitTable_F  $logName | awk '{sum+=$3}END {print "SplitTable_F = " sum/'$iter'}'
grep SplitTable_B  $logName | awk '{sum+=$3}END {print "SplitTable_B = " sum/'$iter'}'

grep Reshape_F  $logName | awk '{sum+=$3}END {print "Reshape_F = " sum/'$iter'}'
grep Reshape_B  $logName | awk '{sum+=$3}END {print "Reshape_B = " sum/'$iter'}'


grep LookupTable_F  $logName | awk '{sum+=$3}END {print "LookupTable_F = " sum/'$iter'}'
grep LookupTable_B  $logName | awk '{sum+=$3}END {print "LookupTable_B = " sum/'$iter'}'

grep Replicate_F  $logName | awk '{sum+=$3}END {print "Replicate_F = " sum/'$iter'}'
grep Replicate_B  $logName | awk '{sum+=$3}END {print "Replicate_B = " sum/'$iter'}'

grep Sum_F  $logName | awk '{sum+=$3}END {print "Sum_F = " sum/'$iter'}'
grep Sum_B  $logName | awk '{sum+=$3}END {print "Sum_B = " sum/'$iter'}'

grep Optim         $logName | awk '{sum+=$3}END {print "Optim        = " sum/'$iter'}'
grep dataloader    $logName | awk '{sum+=$3}END {print "dataloader   = " sum/'$iter'}'
grep copyTensorTable $logName | awk '{sum+=$3}END {print "copyTensorTable = " sum/'$iter'}'

