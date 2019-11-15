for i in 0 1
do
    for j in 0 1
    do
        for name in resnet18_996
            #densenet161_224
            #resnet18_996
            #resnet18_448 seresnet50_448 inception_v4_448
        do
            for data in test val
            do
            echo $name $i $j $data
           # python inferenceMain.py $name $i $j -1 1 $data
            python inferenceMain.py $name $i $j 9_14 1 $data
            done
        done
    done
done
