# circadian_analysis

This repository is for Python codes to analyze circadian time siries.

## Dependency

python3: (worked in 3.7.0)  
library: check `Pipfile` for detail

## Usage
- Install Python
- Donload or git clone this repository
- Run Python script e.g. `python3 "xxxx.py"`

## Papers

- Detection of uncoupled circadian rhythms in individual cells of *Lemna minor* using a dual-color bioluminescence monitoring system  
  Emiri Watanabe, Minako Isoda, Tomoaki Muranaka, Shogo Ito, and Tokitaka Oyama  
  *Plant and Cell Physiology*, Volume 62, Issue 5, May 2021, Pages 815–826, https://doi.org/10.1093/pcp/pcab037
    
  Python file (modules) used in this paper
  - `calculate_min_img.py`: Used for managing image files captured in dual-color monitoring
  - `data_norm` and `cos_fit` in `analyser.FFT_nlls`: Used for estimating period by Fast Fourier transform–nonlinear least squares (FFT-NLLS)
  - `phase_analysis` in `analyser.peak_analysis`: Used for estimating peak time by a local quadratic curve fitting. Parameters were set as follow.  
  `avg=1, dt=60, p_range=12, f_avg=1, f_range=9, offset=0, time=False, r2_cut=False, min_tau=16, max_tau=32`

  
- An endogenous basis for synchronization manners of the circadian rhythm in proliferating Lemna minor plants  
  Kenya Ueno, Shogo Ito, Tokitaka Oyama  
  bioRxiv 2021.02.09.430421, https://doi.org/10.1101/2021.02.09.430421
## Link to laboratory website
[京都大学 理学研究科 植物学教室 形態統御学分科 時間生物学グループ（小山研）](http://cosmos.bot.kyoto-u.ac.jp/clock/)
