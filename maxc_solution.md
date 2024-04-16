### Summary
The main idea of my solution is to generate inputs similar to what the annotators see.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3964695%2Fef831958400ea7348dfe6ecebf6c511a%2FScreenshot%20from%202024-04-09%2012-53-16.png?generation=1712681638219917&alt=media)
- The size of each part is configurable.
    - Final models use size 512x512.
    - The width of raw eeg part is 500 for 10s. (Should be enough for sampling 20Hz signals)
    - The height of raw eeg part is controlled by the number of repeat. E.g. 16 channels * 8 repeat = 128 height.
- Removing the EEG spectrogram part may not affect the performance.

Due to a mistake in my HIGH vote CV implementation (accidentally included some low votes samples), I think I didn't fully optimize the performance in the last week.
- The best score of my part with ensemble (3 models) is Private 0.295527/Public 0.247586. 
- The best single model score: Private=0.303614, Public = 0.250533

### EEG Preprocessing 
After checking the provided EEG pattern images many times, I found that the provided raw EEGs look much cleaner. It was much easier to see the pattern in the provided image than my first few raw EEG plots.

It turns out that simply applying a bandpass filter can make the raw EEGs look much closer to the provided examples.
I also noticed that the scipy `sosfilt` is preferred over the commonly used `lfilter` according to the [document](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html) and the `sosfiltfilt` can reduce the effect of phase shift. 

- https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter/48677312#48677312
- Thanks to @gunesevitan for sharing this [discussion](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/483839)

The red boxes in this diagram show the possible phase shifts:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3964695%2Ff2e4493929bebfd7149582d9e23e8c0b%2FScreenshot%20from%202024-04-09%2012-56-41.png?generation=1712681817517725&alt=media)


### Training
- One stage training with lower weight for LOW vote samples.
    - Monitor CV score for both All and HIGH vote samples under different weight ratios.
    - The final weight selected is 0.25 for low vote samples (<=7) for best ALL and HIGH vote CV score.
    - The weight is very close to (3.04/14.29 ~= 0.21), which is `average # LOW votes / average # HIGH votes`.
- Exponential Moving Average (EMA) did well in stabilizing the validation loss during training. 

### Augmentation
- Flip Left and Right components
- Horizontal flip for spectrograms
- Time and Frequency mask for the spectrograms
- Time and Channel mask for the raw EEG. (Randomly hide 4-6 channels in the raw EEG)
- Time shift of 10s Raw EEG
- Raw EEG * -1 
- Small random contrast adjustment for the raw eeg.

Test time augmentation(TTA):
- Flip Left and Right components.

### Model
Pretrained backbones from TIMM with different pooling layers.
- Head: LayerNorm -> Dropout -> FC 
- Models used in final submission:
    - maxvit_small_tf_512, 
    - maxvit_tiny_tf_512,
    - convnext_small
- Average pooling performs the best in my experiments.

