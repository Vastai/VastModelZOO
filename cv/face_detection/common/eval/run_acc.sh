#! /bin/bash
file_folder='/opt/test-script/test_results/models_fixed/Acc/retinaface'
echo "start process folder: $file_folder"
files=`ls $file_folder`
echo "list files:\n$files"
if [ -d ${file_folder}/acc ]
then
    echo 'exists'
else
    mkdir ${file_folder}/acc
fi

for file in ${files[@]}
do
python3 evaluation.py -p ${file_folder}/$file | tee ${file_folder}/acc/$file.txt

done


exit

