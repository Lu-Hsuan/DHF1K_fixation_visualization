# DHF1K_fixation_visualization

## Directory information
* /video            : DHF1K Video data
* /eportdata_train  : DHF1K Fixation data 
* /out_video        : visualization output video
* /record_npy       : DHF1K Fixation data to numpy .npy

## Visualization process
首先讀取DHF1K資料集的注視點資料，共17位受測者，處理成numpy array。
接下來對17位受測者進行分群處理，使用MeanShift分群演算法，計算出注視點的群體，從中挑取最大群作為顯示。
使用bezier插值，實現注視點之移動動畫。
黃色點為群中心。

## Example Visualization
![c1](https://github.com/Lu-Hsuan/DHF1K_fixation_visualization/blob/master/example_img/img1.PNG)

![c2](https://github.com/Lu-Hsuan/DHF1K_fixation_visualization/blob/master/example_img/img2.PNG)

## Video
please Download out_video example
