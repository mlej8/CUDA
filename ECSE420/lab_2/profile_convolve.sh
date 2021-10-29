#! /bin/bash

for i in {1..3}
do
    for num_threads in "1" "4" "8" "16" "64" "128" "256" "512" "1024"
        do
            f_name=test_image_"$i"_num_threads_"$num_threads"
            nsys profile --output reports/"$f_name" ./build/convolve ./Test/Test_$i.png ./Output/Test_$i.png $num_threads
            nsys stats reports/"$f_name".qdrep  >> reports/"$f_name".txt
            if [ $i -eq 1 ]
            then
                ./validate.sh
            fi
            
        done
done