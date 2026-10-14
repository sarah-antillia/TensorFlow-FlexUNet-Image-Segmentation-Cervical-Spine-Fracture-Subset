<h2>TensorFlow-FlexUNet-Image-Segmentation-Cervical-Spine-Fracture-Subset (2026/10/14)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

This is the first experiment of Image Segmentation for <b>Cervical Spine Fracture Subset</b>
 based on
our <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">TensorFlowFlexUNet Model</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and a 512x512-pixel PNG
 <a href="https://drive.google.com/file/d/1TDUzyOvsFTeW2OKaxk5-tDWVzQMUVsOW/view?usp=sharing">
Cervical-Spine-Fracture-ImageMask-Subset.zip</a> (<a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0: Public Domain</a>
), 
which was derived by us from the Kaggle  
<br><br>
<a href="https://www.kaggle.com/datasets/akshat0912/png-data-for-cervical-spine-fracture-segmentation">
<b>PNG Data for Cervical Spine Fracture Segmentation</b>
</a> by Akshat Verma.
<br>
<hr>
<b>Actual Image Segmentation for Cervical Spine Fracture Images of 512x512 pixels</b><br>
As shown below, the inferred masks resemble the ground-truth masks. <br>
<br>
<table>
<tr>
<th width="320" height="auto">Input: image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/1542_385.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/1542_385.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/1542_385.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/1868_572.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/1868_572.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/1868_572.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/16919_301.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/16919_301.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/16919_301.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from the following Kaggle:<br><br>
<a href="https://www.kaggle.com/datasets/akshat0912/png-data-for-cervical-spine-fracture-segmentation">
<b>PNG Data for Cervical Spine Fracture Segmentation</b>
</a> by Akshat Verma.
<br><br>
This PNG Dataset was derived from 
<a href="https://pubs.rsna.org/doi/10.1148/ryai.230034"><b>The RSNA Cervical Spine Fracture CT Dataset</b></a>
<br><br>
For more information, please refer to the Kaggle website<br><br>
<a href="https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/">
<b>RSNA 2022 Cervical Spine Fracture Detection</b></a>
<br><br>
<b>Context</b><br>
RSNA has teamed with the American Society of Neuroradiology (ASNR) and the American Society of 
Spine Radiology
 (ASSR) to conduct an AI challenge competition exploring whether artificial intelligence can be used to aid 
 in the detection and localization of cervical spine fractures.
<br><br>
To create the ground truth dataset, the challenge planning task force collected imaging data sourced from 
twelve sites on six continents, including approximately 3,000 CT studies. Spine radiology specialists 
from the ASNR and ASSR provided expert image level annotations these studies to indicate the presence, 
vertebral level and location of any cervical spine fractures.
<br><br>
In this challenge competition, you will try to develop machine learning models that match the radiologists' 
performance in detecting and localizing fractures to the seven vertebrae that comprise the cervical spine. 
Winners will be recognized at an event during the RSNA 2022 annual meeting.
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0: Public Domain</a>
<br>
<br>
<h3>
2 Cervical-Spine-Fracture-Subset Ultrasound ImageMask Dataset
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this Cervical-Spine-Fracture-Subset Segmentation model,
 please download the dataset from Google Drive  
 <a href="https://drive.google.com/file/d/1TDUzyOvsFTeW2OKaxk5-tDWVzQMUVsOW/view?usp=sharing">
Cervical-Spine-Fracture-Subset-ImageMask-Subset.zip</a>. 
Expand the downloaded ImageMaskDataset and put it under the <b>./dataset</b> folder.
<br>
<pre>
./dataset
└─Cervical-Spine-Fracture-Subset
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
         ├─images
         └─masks
</pre>
<br>
<b>Cervical-Spine-Fracture-Subset Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/Cervical-Spine-Fracture-Subset_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images in the training and valid datasets is large enough to use for the
 training set of our segmentation model.
<br>
<h3>2.2 Derivation of ImageMask Dataset</h3>
The folder structure of the original dataset <a href="https://www.kaggle.com/datasets/akshat0912/png-data-for-cervical-spine-fracture-segmentation">
<b>PNG Data for Cervical Spine Fracture Segmentation</b>
</a> is as follows.
<pre>
./Cervical_Spine_Fracture
 ├─test_png
 │   ├─images ( 5,663 files)
 │   │
 │   └─mask   ( 5,663 files)
 │ 
 └─train_png   
      ├─images (24,169 files)
      │
      └─masks  (24,169 files)
</pre>
In this experiment, we generated a one-third subset of the  <b>train_png</b> for simplicity. <br>
<br>
<h3>2.3 Train Sample Images and Masks</h3>
<b>Train_sample_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_sample_masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained the Cervical-Spine-Fracture-Subset TensorFlowFlexUNet model using the following
<a href="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
This runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16 </b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
; You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_focal_dice_loss" and <a href="./src/dice_coef_multiclass.py">"dice_coef_hybrid"</a>.<br>
<pre>
[model]
loss           = "categorical_focal_dice_loss"
metrics        = ["dice_coef_hybrid"]
hybrid_alpha   = 1.6
hybrid_beta    = 0.4
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled the learning_rate_reducer callback and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with the patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
Specified RGB color map dict for Cervical-Spine-Fracture-Subset 2 classes.<br>
<pre>
[mask]
mask_datatype= "categorized"
mask_file_format = ".png"
; Cervical-Spine-Fracture-Subset RGB color map dict for 1+1 classes.
;      Background: black, Fracture: white
rgb_map = {(0,0,0):0,(255,255,255):1,}

</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch change, the inference procedure can be called
 for 6 images in the <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our FlexUNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 14,15,16)</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 29,30,31)</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 31 by EarlyStoppingCallback.<br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/train_console_output_at_epoch31.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset</b> folder,<br>
and run the following bat file to evaluate the TensorFlowUNet model for Cervical-Spine-Fracture-Subset.<br>
<pre>
./2.evaluate.bat
</pre>
This runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/evaluate_console_output_at_epoch31.png" width="1024" height="auto">
<br><br>Image-Segmentation-Cervical-Spine-Fracture-Subset

<a href="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_focal_dice_loss) on this Cervical-Spine-Fracture-Subset/test was low, and dice_coef_hybrid was high, as shown below.
<br>
<pre>
categorical_focal_dice_loss,0.0117
dice_coef_hybrid,0.9796
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset</b> folder
and run the following bat file to infer segmentation regions for images using the trained TensorFlowUNet model for Cervical-Spine-Fracture-Subset.<br>
<pre>
./3.infer.bat
</pre>
This runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Cervical-Spine-Fracture-Subset MRI Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks.<br>
<br>
<table>
<tr>
<th width="320" height="auto">Image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/10633_137.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/10633_137.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/10633_137.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/10633_360.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/10633_360.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/10633_360.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/14267_82.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/14267_82.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/14267_82.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/16919_317.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/16919_317.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/16919_317.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/19388_382.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/19388_382.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/19388_382.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/images/20928_215.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test/masks/20928_215.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Spine-Fracture-Subset/mini_test_output/20928_215.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Automated Multi-Task Deep Learning Framework for Cervical Spine Fracture Segmentation</b><br>
Shengjie Gou, Anlin Luo, Yingping Li<br>
<a href="https://dl.acm.org/doi/epdf/10.1145/3784929.3784933">https://dl.acm.org/doi/epdf/10.1145/3784929.3784933</a>
<br><br>
<b>2. An efficient deep learning based approach for automated 
identification of cervical vertebrae fracture as a clinical support aid</b><br>
Singh, M., Tripathi, U., Patel, K.K. et al.<br>
<a href="https://www.nature.com/articles/s41598-025-10448-6">https://www.nature.com/articles/s41598-025-10448-6</a>
<br><br>
<b>3. 3D Cervical Spine Segmentation and Multi-Vertebrae Fracture Detection</b><br>
Swayam<br>
<a href="https://github.com/SwayamInSync/3D-Cervical-Spine-Segmentation-and-Multi-Vertebrae-Fracture-Detection">
https://github.com/SwayamInSync/3D-Cervical-Spine-Segmentation-and-Multi-Vertebrae-Fracture-Detection</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Bladder-Cancer-MRI-New-Dice-Metric</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Bladder-Cancer-MRI-New-Dice-Metric">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Bladder-Cancer-MRI-New-Dice-Metric</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br><br>
