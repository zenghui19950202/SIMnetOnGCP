# SIMnetOnGCP
先使用 GenerateSIMdataWithPhiThetaLabel.py 这个程序生成SIM的数据集：
  1.需要自己找一些图片放在一个文件夹下，程序会根据这些图片生成训练的数据集。图片文件夹的路径需要赋给 SourceFileDirectory这个变量
  2.同时需要自己提供一个txt的文件，接收数据集的路径和label。  这个txt的路径需要赋给 directory_txt_file 变量
  
使用train.py 进行net的训练
