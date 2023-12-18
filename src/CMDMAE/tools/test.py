from get_audio_from_video import load_audio_from_video
from to_spec import to_spec
from griffin_lim import griffin_lim
import librosa
import numpy as np
from scipy.io.wavfile import write

audio = load_audio_from_video(
    path_video=r"D:\These\data\Audio-Visual\voxceleb\test_\parta\id01460\Ct1XMedcGHk\00072.mp4", sr=16000)

wav = librosa.to_mono(audio.transpose())
wav = wav / np.max(np.abs(wav))

win_length = int(64e-3 * 16000)
spec_parameters = dict(n_fft=1024,
                       hop=int(0.625 * win_length),
                       win_length=win_length)
spec, _ = to_spec(wav, spec_parameters=spec_parameters)
print(spec.shape)
rec = griffin_lim(spec)


write(filename="rec.wav", rate=16000, data=rec)
write(filename="real.wav", rate=16000, data=wav)

