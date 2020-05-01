data preprocessing:

  1) create dataset
    
    reads manual annotations
    
    segments audio files and 
    
    computes spectrograms per segment
    
    saves dataset: numpy array: [[segment_id],[spectrogram (n_mels, n_timeframes)], [label_matrix (8*n_timeframes)]]
  
  2) adapt dataset to task
  
  3) split dataset and normalize
  
  4)[OPT] Analysis dataset

Train

  1) train.py
    
 
predict

  1) apply_model notebook
  
