# rtsr
recurrently trained super-resolution

if you want to run with full data 
  1. make 100 image crops per a DIV2K train image
    -> positioning them ./dataset/DIV2K_train_HR/images/train_crop_sr
  2. compress them
    -> positioning them ./dataset/DIV2K_train_HR/images/compressed_train_crop_png
  3. set image you want to SR
    -> positioning them ./dataset/DIV2K_train_HR/images/valid_sr

runing
  sh ./run_all.sh rtsr
  rtsr : can replace (srcnn / srresnet / vdsr / edsr)

result 
   ./output
