mkdir ./dataset/DIV2K_train_HR/images/compressed_train_crop_png

for file in ./dataset/DIV2K_train_HR/images/compressed_train_crop_jpg/*.jpg
do
   name=${file##*/}
   base=${name%.jpg}
   ffmpeg -i $file -qscale:v 6 "./dataset/DIV2K_train_HR/images/compressed_train_crop_png/$base.png"
done


