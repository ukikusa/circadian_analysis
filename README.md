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

- An endogenous basis for synchronisation characteristics of the circadian rhythm in proliferating *Lemna minor* plants
  Kenya Ueno, Shogo Ito, Tokitaka Oyama
  *New Phytologist*, Volume233, Issue5, March 2022, Pages 2203-2215, [https://doi.org/10.1111/nph.17925](https://doi.org/10.1111/nph.17925)

  Python file (modules) used in this paper
  - `img_circadian_analysis` in `analyser.phase_from_img`: Used for rhythmic analysis in the fronds (Fig2, 3). Parameters were set as follow. `img_circadian_analysis(folder, avg=3, calc_range=[start, start + 24 * 3, dt=60, offset=offset, save=True, background=background, tau_range=[16, 32])`

## Link to laboratory website

[京都大学 理学研究科 植物学教室 形態統御学分科 時間生物学グループ（小山研）](http://cosmos.bot.kyoto-u.ac.jp/clock/)

## notes

There is little documentation for this code. Functions not used in the paper may have bugs in their arguments. Ueno plans to improve the code as a hobby. Please contact [@kenya_ueno](https://twitter.com/kenya_ueno) if you have an urgent need for a function.

## Citation

    @article{ueno2022endogenous,
      title={An endogenous basis for synchronisation characteristics of the circadian rhythm in proliferating Lemna minor plants},
      author={Ueno, Kenya and Ito, Shogo and Oyama, Tokitaka},
      journal={New Phytologist},
      volume={233},
      number={5},
      pages={2203--2215},
      year={2022},
      publisher={Wiley Online Library}
    }