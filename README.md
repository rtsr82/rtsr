# rtsr
recurrently trained super-resolution

you can do with out setting a data and just doing test with sample : (1/20 of DIV2K)

if you want to run with full data :

  1. make 100 image crops per a DIV2K train image
    -> positioning them ./dataset/DIV2K_train_HR/images/train_crop_sr
  2. compress them
    -> positioning them ./dataset/DIV2K_train_HR/images/compressed_train_crop_png
  3. set image you want to SR
    -> positioning them ./dataset/DIV2K_train_HR/images/valid_sr

runing : 
  sh ./run_all.sh (rtsr / srcnn / srresnet / vdsr / edsr)
  
  ex) sh ./run_all.sh rtsr

result : 
   ./output
