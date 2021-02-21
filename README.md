# Neural-Speech-Dereverberation
Machine and Deep Learning models for speech dereverberation

## Data
- LibriSpeech for speech audio files
- Omni and MARDY dataset for Room Impulse Responses (RIRs)
- BUT Speech@FIT Reverb Database for retransmitted data. Available: \url{https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database}


## Models

- MLP 
- LSTM based model 
- FD-NDLP (WPE + frequency domain). 
  Implementation taken from https://github.com/helianvine/fdndlp
- U-net for speech dereverberation [1]
- GAN training with U-net generator [1]


## References

[1] Ori Ernst, Shlomo E. Chazan, Sharon Gannot and Jacob Goldberger, "Speech Dereverberation Using Fully Convolutional Networks". Faculty of Engineering, Bar-Ilan University, 3 Apr, 2019.