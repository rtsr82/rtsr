 mkdir ./dataset/DIV2K_train_HR/images/valid_sr/crop_4x2
 for file in ./dataset/DIV2K_train_HR/images/valid_sr/1/*.png
 do
    name=${file##*/}
    python crop_4x2.py --input_image $file --output_filename "./dataset/DIV2K_train_HR/images/valid_sr/crop_4x2/$name"
done
 
