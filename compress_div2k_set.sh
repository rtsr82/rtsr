mkdir ./dataset/DIV2K_train_HR/images/compressed_train_crop_jpg

for file in ./dataset/DIV2K_train_HR/images/train_crop_sr/1/*.png
do
    name=${file##*/}
    base=${name%.png}
    ffmpeg -i $file -qscale:v 6 "./dataset/DIV2K_train_HR/images/compressed_train_crop_jpg/$base.jpg"
done



