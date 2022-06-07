@REM ##===== Testing on image_test =====##
@REM ## Small size ##
@REM python test.py --model_path model_trained/net_F16B2E2_epoch_15.pth ^
@REM                --input_image_path image_test/LR_zebra_test.png ^
@REM                --output_image_path result/F16B2E2_zebra_test.png ^
@REM                --compare_image_path reference/F16B2E2_zebra_test_golden.png ^
@REM                --cuda

@REM ## Large size ##
python test.py --model_path model_trained/net_F32B8E4_epoch_120.pth ^
               --input_image_path image_test/LR_zebra_test.png ^
               --output_image_path result/F32B8E4_zebra_test.png ^
               --compare_image_path reference/F32B8E4_zebra_test_golden.png ^
               --cuda

@REM ##===== Testing on image_hidden =====##
@REM set obj=LR_crossing
@REM set obj=LR_horse
@REM set obj=LR_panda
@REM set obj=LR_pups

@REM ## Small size ##
@REM python test.py --model_path model_trained/net_F16B2E2_epoch_15.pth ^
@REM                --input_image_path image_hidden/%obj%.png ^
@REM                --output_image_path result/F16B2E2_%obj%.png ^
@REM                --cuda

@REM ## Large size ##
@REM python test.py --model_path model_trained/net_F32B8E4_epoch_120.pth ^
@REM                --input_image_path image_hidden/%obj%.png ^
@REM                --output_image_path result/F32B8E4_%obj%.png ^
@REM                --cuda