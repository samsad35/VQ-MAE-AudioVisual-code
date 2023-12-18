import os.path
import os
from tkinter import *
from tkinter.filedialog import *
from tkinter import messagebox
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import warnings
import hydra
from hydra.experimental import compose, initialize
from ..model import VQVAE, SpeechVQVAE, CMDMAE
from ..tools import size_model, read_video_decord, to_spec
import librosa
import torchvision.transforms as transforms
import kornia as K
import librosa.display as display
import math
from einops import rearrange
from scipy.io.wavfile import write
import sounddevice as sd


class InterfaceVQMAE:
    def __init__(self):
        self.master = Tk()
        self.master.title("VQ-MAE-AudioVisual 0.1")
        self.master.geometry("1600x1000")
        # self.master.wm_iconbitmap(r'sf_vae\sf_vae\Gui\icon.ico')
        self.master.resizable(width=False, height=False)
        self.device = torch.device("cuda")

        # -- -- .Json
        self.button = Button(self.master, text="Config file", command=self.get_config, activebackground='green')
        self.button.place(relx=0.48, rely=0.05, anchor=CENTER)

        self.resolution = IntVar()
        Checkbutton(self.master, text='Resolution',
                    variable=self.resolution,
                    onvalue=1,
                    offvalue=0).place(relx=0.30, rely=0.05, anchor=CENTER)

        # -- --
        self.zss, self.zds, self.zaudio, self.zvisual = StringVar(), StringVar(), StringVar(), StringVar()

    def get_config(self):
        self.path_config = askdirectory(title="Open the config directory")
        self.path_config_relative = os.path.relpath(self.path_config,
                                                    r"D:\These\Git\Mulitimodal Dynamical Masked Auto-Encoder\src\CMDMAE\interface")
        print(self.path_config)
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        # -- -- .Model
        Button(self.master, text="VQ-MAE-AV", command=self.get_mdvae).place(relx=0.56, rely=0.05, anchor=CENTER)

    def get_mdvae(self):
        try:
            initialize(config_path=self.path_config_relative)
            cfg = compose(config_name="config")
        except ValueError:
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            initialize(config_path=self.path_config_relative)
            cfg = compose(config_name="config")
        if self.resolution.get() == 0:
            self.vqvae_1 = VQVAE(**cfg.vqvae_1)
            self.vqvae_1.load(path_model=r"checkpoint/VQVAE/2023-1-25/10-27/model_checkpoint")
            self.vqvae_1.to('cuda')
            size_model(self.vqvae_1, text="vqvae_1")
        else:
            self.vqvae_1 = VQVAE(**cfg.vqvae_1)
            self.vqvae_1.load(path_model=r"checkpoint/VQVAE/2023-3-22/19-2/model_checkpoint")
            self.vqvae_1.to('cuda')
            size_model(self.vqvae_1, text="vqvae_1")

        self.vqvae_2 = SpeechVQVAE(**cfg.vqvae_2)
        self.vqvae_2.load(path_model=r"checkpoint/SPEECH_VQVAE/2023-3-1/9-8/model_checkpoint")
        self.vqvae_2.to('cuda')
        size_model(self.vqvae_2, text="vqvae_2")
        try:
            self.cmdmae = CMDMAE(**cfg.model,
                                 alpha=(1.0, 1.0),
                                 pos_embedding_trained=True,
                                 vqvae_v_embedding=None,
                                 vqvae_a_embedding=None,
                                 # decoder_cross_attention=True,
                                 # encoder_cross_attention=False,
                                 # contrastive=False,
                                 mlp_ratio=2.0)
            size_model(self.cmdmae, text="cmdmae")
            self.cmdmae.load(path_model=f"{os.path.dirname(self.path_config)}/model_checkpoint")
            self.cmdmae.to("cuda")
        except:
            messagebox.showerror(title='Load model @Error', message="Problème dans le modèle, vérifier les paramètres")
        self.build_1()

    def build_1(self):
        # -- Sequence Length --
        self.seq_length = IntVar()
        self.seq_length.set(50)
        entree = Entry(self.master, textvariable=self.seq_length, width=10)
        entree.place(relx=0.11, rely=0.02, anchor=CENTER)
        Label(self.master, text="Sequence length: ").place(relx=0.04, rely=0.02, anchor=CENTER)

        Button(self.master, text="Load data", command=self.get_inputs, activebackground='green').place(relx=0.15,
                                                                                                       rely=0.05,
                                                                                                       anchor=CENTER)
        # -- --
        self.frame_visual = Frame(self.master, height=350, width=500, borderwidth=2, relief=GROOVE)
        self.frame_visual.place(relx=0.15, rely=0.30, anchor=CENTER)

        self.frame_audio = Frame(self.master, height=350, width=500, borderwidth=2, relief=GROOVE)
        self.frame_audio.place(relx=0.15, rely=0.80, anchor=CENTER)

        self.frame_visual_masked = Frame(self.master, height=350, width=500, borderwidth=2, relief=GROOVE)
        self.frame_visual_masked.place(relx=0.50, rely=0.30, anchor=CENTER)

        self.frame_audio_masked = Frame(self.master, height=350, width=500, borderwidth=2, relief=GROOVE)
        self.frame_audio_masked.place(relx=0.50, rely=0.80, anchor=CENTER)

        self.frame_visual_recon = Frame(self.master, height=350, width=500, borderwidth=2, relief=GROOVE)
        self.frame_visual_recon.place(relx=0.85, rely=0.30, anchor=CENTER)

        self.frame_audio_recon = Frame(self.master, height=350, width=500, borderwidth=2, relief=GROOVE)
        self.frame_audio_recon.place(relx=0.85, rely=0.80, anchor=CENTER)

    @staticmethod
    def load_wav(file: str):
        wav, sr = librosa.load(path=file, sr=16000)
        wav = librosa.to_mono(wav.transpose())
        wav = wav / np.max(np.abs(wav))
        return wav

    def get_inputs(self):
        self.path_video = askopenfilename(title="Open AV file", filetypes=[('mp4 files', '.mp4'), ('all files', '.*')])
        if self.resolution.get() == 0:
            len = 50
            transform = transforms.Compose(
                [transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                 transforms.Resize(128),
                 transforms.CenterCrop(96)])
        else:
            len = 25
            transform = transforms.Compose(
                [transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                 transforms.Resize(224),
                 transforms.CenterCrop(192)])
        video = read_video_decord(self.path_video).transpose((0, 3, 1, 2))[:len]
        audio = self.load_wav(self.path_video)
        win_length = int(64e-3 * 16000)
        spec_parameters = dict(n_fft=1024,
                               hop=int(0.625 / 2 * win_length),
                               win_length=win_length)
        audio, self.phase = to_spec(audio, spec_parameters)
        self.phase = self.phase[:, :len*2]
        audio = torch.from_numpy((audio ** 2).transpose()).type(torch.FloatTensor).unsqueeze(1)[:len*2]

        video = torch.from_numpy(video) / 255.0

        video = transform(video)

        audio = audio.to("cuda")
        self.indices_audio = self.vqvae_2.get_codebook_indices(audio)

        video = video.to("cuda")
        self.indices_video = self.vqvae_1.get_codebook_indices(video)

        self.ratio = DoubleVar()
        self.ratio.set(50)
        self.slider = Scale(self.master, from_=0, to=100, length=350, orient='horizontal',
                            variable=self.ratio).place(relx=0.50, rely=0.52, anchor=CENTER)
        Button(self.master, text="Run", command=self.masking, bg="khaki4").place(relx=0.63, rely=0.05, anchor=CENTER)
        Button(self.master, text="Save", command=self.save, bg="green").place(relx=0.70, rely=0.05, anchor=CENTER)

        fig = self.plot_images(self.indices_video, show=False, vqvae=self.vqvae_1)
        self.canvas_1 = FigureCanvasTkAgg(fig, master=self.frame_visual)
        self.canvas_1.get_tk_widget().place(relx=0.5, rely=0.53, anchor=CENTER)
        self.canvas_1.draw()

        fig = self.plot_spectrogram(self.indices_audio, show=False, vqvae=self.vqvae_2)
        self.canvas_2 = FigureCanvasTkAgg(fig, master=self.frame_audio)
        self.canvas_2.get_tk_widget().place(relx=0.5, rely=0.47, anchor=CENTER)
        self.canvas_2.draw()

        Button(self.frame_visual, text="zoom", command=self.zoom_images, activebackground='green').place(relx=0.74,
                                                                                                         rely=0.94,
                                                                                                         anchor=CENTER)
        Button(self.frame_audio, text="zoom", command=self.zoom_spectrogram, activebackground='green').place(relx=0.74,
                                                                                                             rely=0.94,
                                                                                                             anchor=CENTER)

        Button(self.frame_visual, text="animation", command=self.animation_images, activebackground='green').place(
            relx=0.26,
            rely=0.94,
            anchor=CENTER)
        Button(self.frame_audio, text="play", command=self.play_spectrogram, activebackground='green').place(
            relx=0.26,
            rely=0.94,
            anchor=CENTER)

    def animation_images(self):
        self.save_animation(self.indices_video, vqvae=self.vqvae_1)

    def animation_images_masked(self):
        self.save_animation(self.images_mask_1[0], vqvae=self.vqvae_1)

    def animation_images_recons(self):
        self.save_animation(self.indices_visual_recon[0], vqvae=self.vqvae_1)

    def zoom_images(self):
        fig = self.plot_images(self.indices_video, show=True, vqvae=self.vqvae_1, fig_size=(7.5, 6))

    def zoom_spectrogram(self):
        fig = self.plot_spectrogram(self.indices_audio, show=True, vqvae=self.vqvae_2, fig_size=(7.5, 6))

    def zoom_images_masked(self):
        fig = self.plot_images(self.images_mask_1[0], show=True, vqvae=self.vqvae_1, fig_size=(7.5, 6))

    def zoom_spectrogram_masked(self):
        fig = self.plot_spectrogram(self.audio_mask[0], show=True, vqvae=self.vqvae_2, fig_size=(7.5, 6))

    def zoom_images_recons(self):
        fig = self.plot_images(self.indices_visual_recon[0], show=True, vqvae=self.vqvae_1, fig_size=(7.5, 6))

    def zoom_spectrogram_recons(self):
        fig = self.plot_spectrogram(self.indices_audio_recon[0], show=True, vqvae=self.vqvae_2, fig_size=(7.5, 6))

    def play_spectrogram_recons(self):
        self.save_wav(self.indices_audio_recon[0], vqvae=self.vqvae_2, phase=self.phase, save=None, play=True)

    def play_spectrogram_masked(self):
        self.save_wav(self.audio_mask[0], vqvae=self.vqvae_2, phase=self.phase, save=None, play=True)

    def play_spectrogram(self):
        self.save_wav(self.indices_audio, vqvae=self.vqvae_2, phase=self.phase, save=None, play=True)

    def run(self):
        None

    def save(self):
        path_save = askdirectory()
        self.plot_images(self.indices_video, show=False, vqvae=self.vqvae_1,
                         fig_size=(7.5, 6), save=f"{path_save}/images-original_ratio-{self.ratio.get()}.svg")
        self.plot_images(self.images_mask_1[0], show=False, vqvae=self.vqvae_1,
                         fig_size=(7.5, 6), save=f"{path_save}/images-masked_ratio-{self.ratio.get()}.svg")
        self.plot_images(self.indices_visual_recon[0], show=False, vqvae=self.vqvae_1,
                         fig_size=(7.5, 6),
                         save=f"{path_save}/images-reconstructed_ratio-{self.ratio.get()}.svg")

        self.save_animation(self.indices_video, vqvae=self.vqvae_1, fig_size=(3, 3),
                            video_name=f"{path_save}/images-original_ratio-{self.ratio.get()}.mp4")
        self.save_animation(self.images_mask_1[0], vqvae=self.vqvae_1, fig_size=(3, 3),
                            video_name=f"{path_save}/images-masked_ratio-{self.ratio.get()}.mp4")
        self.save_animation(self.indices_visual_recon[0], vqvae=self.vqvae_1, fig_size=(3, 3),
                            video_name=f"{path_save}/images-reconstructed_ratio-{self.ratio.get()}.mp4")

        self.plot_spectrogram(self.indices_audio, show=False, vqvae=self.vqvae_2, fig_size=(7.5, 6),
                              save=f"{path_save}/spectrogram-original_ratio-{self.ratio.get()}.svg")
        self.plot_spectrogram(self.audio_mask[0], show=False, vqvae=self.vqvae_2, fig_size=(7.5, 6),
                              save=f"{path_save}/spectrogram-masked_ratio-{self.ratio.get()}.svg")
        self.plot_spectrogram(self.indices_audio_recon[0], show=False, vqvae=self.vqvae_2, fig_size=(7.5, 6),
                              save=f"{path_save}/spectrogram-reconstructed_ratio-{self.ratio.get()}.svg")

        self.save_wav(self.indices_audio, vqvae=self.vqvae_2, phase=self.phase,
                      save=f"{path_save}/spectrogram-original_ratio-{self.ratio.get()}.wav")
        self.save_wav(self.audio_mask[0], vqvae=self.vqvae_2, phase=self.phase,
                      save=f"{path_save}/spectrogram-masked_ratio-{self.ratio.get()}.wav")
        self.save_wav(self.indices_audio_recon[0], vqvae=self.vqvae_2, phase=self.phase,
                      save=f"{path_save}/spectrogram-reconstructed_ratio-{self.ratio.get()}.wav")

    def masking(self):
        tube_visual = self.to_tube(self.indices_video[None])
        tube_audio = self.to_tube(self.indices_audio[None], input_image=False)
        ratio = torch.tensor(self.ratio.get() / 100)
        self.indices_visual_recon, mask_v, self.indices_audio_recon, mask_a = \
            self.cmdmae(tube_visual, tube_audio, ratio=(ratio, 1 - ratio))

        _, self.indices_visual_recon = torch.max(self.indices_visual_recon.data, -1)
        self.indices_visual_recon = (self.indices_visual_recon * mask_v + tube_visual * (~mask_v.to(torch.bool))).type(
            torch.int64)
        images_mask_1 = (tube_visual * (~mask_v.to(torch.bool))).type(torch.int64)
        self.indices_visual_recon = self.inverse_tuple(self.indices_visual_recon, input_image=True, size_patch=4)
        self.images_mask_1 = self.inverse_tuple(images_mask_1, input_image=True, size_patch=4)
        # self.images_mask_1[self.images_mask_1 == 0] = 2
        fig = self.plot_images(self.images_mask_1[0], show=False, vqvae=self.vqvae_1)
        self.canvas_1 = FigureCanvasTkAgg(fig, master=self.frame_visual_masked)
        self.canvas_1.get_tk_widget().place(relx=0.5, rely=0.53, anchor=CENTER)
        self.canvas_1.draw()

        fig = self.plot_images(self.indices_visual_recon[0], show=False, vqvae=self.vqvae_1)
        self.canvas_1 = FigureCanvasTkAgg(fig, master=self.frame_visual_recon)
        self.canvas_1.get_tk_widget().place(relx=0.5, rely=0.53, anchor=CENTER)
        self.canvas_1.draw()

        _, self.indices_audio_recon = torch.max(self.indices_audio_recon.data, -1)
        self.indices_audio_recon = (self.indices_audio_recon * mask_a + tube_audio * (~mask_a.to(torch.bool))).type(
            torch.int64)
        audio_mask = (tube_audio * (~mask_a.to(torch.bool))).type(torch.int64)
        self.indices_audio_recon = self.inverse_tuple(self.indices_audio_recon, input_image=False, size_patch=4)
        self.audio_mask = self.inverse_tuple(audio_mask, input_image=False, size_patch=4)
        # self.audio_mask[self.audio_mask == 0] = 2
        fig = self.plot_spectrogram(self.audio_mask[0], show=False, vqvae=self.vqvae_2)
        self.canvas_1 = FigureCanvasTkAgg(fig, master=self.frame_audio_masked)
        self.canvas_1.get_tk_widget().place(relx=0.5, rely=0.47, anchor=CENTER)
        self.canvas_1.draw()

        fig = self.plot_spectrogram(self.indices_audio_recon[0], show=False, vqvae=self.vqvae_2)
        self.canvas_1 = FigureCanvasTkAgg(fig, master=self.frame_audio_recon)
        self.canvas_1.get_tk_widget().place(relx=0.5, rely=0.47, anchor=CENTER)
        self.canvas_1.draw()

        Button(self.frame_visual_recon, text="zoom", command=self.zoom_images_recons, activebackground='green').place(
            relx=0.74,
            rely=0.94,
            anchor=CENTER)
        Button(self.frame_visual_recon, text="animation", command=self.animation_images_recons,
               activebackground='green').place(
            relx=0.26,
            rely=0.94,
            anchor=CENTER)

        Button(self.frame_audio_masked, text="zoom", command=self.zoom_spectrogram_masked,
               activebackground='green').place(
            relx=0.74,
            rely=0.94,
            anchor=CENTER)
        Button(self.frame_audio_masked, text="play", command=self.play_spectrogram_masked,
               activebackground='green').place(
            relx=0.26,
            rely=0.94,
            anchor=CENTER)
        Button(self.frame_audio_recon, text="zoom", command=self.zoom_spectrogram_recons,
               activebackground='green').place(
            relx=0.74,
            rely=0.94,
            anchor=CENTER)
        Button(self.frame_audio_recon, text="play", command=self.play_spectrogram_recons,
               activebackground='green').place(
            relx=0.26,
            rely=0.94,
            anchor=CENTER)
        Button(self.frame_visual_masked, text="zoom", command=self.zoom_images_masked, activebackground='green').place(
            relx=0.74,
            rely=0.94,
            anchor=CENTER)
        Button(self.frame_visual_masked, text="animation", command=self.animation_images_masked,
               activebackground='green').place(
            relx=0.26,
            rely=0.94,
            anchor=CENTER)

    @staticmethod
    def to_tube(input, input_image: bool = True, size_patch=4):
        if input_image:
            c1 = c2 = int(math.sqrt(input.shape[-1]) / size_patch)
            t1 = input.shape[1] // 5
            input = rearrange(input, 'b (t1 t2) (c1 l1 c2 l2) -> b (t1 c1 c2) (l1 l2 t2)',
                              t1=t1, t2=5, c1=c1, c2=c2, l1=size_patch, l2=size_patch)
        else:
            c1 = int(input.shape[-1] / size_patch)
            t1 = input.shape[1] // 10
            input = rearrange(input, 'b (t1 t2) (c1 l1) -> b (t1 c1) (l1 t2)', t1=t1, t2=10, c1=c1, l1=size_patch)
        return input

    def inverse_tuple(self, input, input_image: bool = True, size_patch=4):
        if self.resolution.get() == 0:
            channel = 6
            t1 = 10
        else:
            channel = 12
            t1 = 5
        if input_image:
            input = rearrange(input, 'b (t1 c1 c2) (l1 l2 t2) -> b (t1 t2) (c1 l1 c2 l2)',
                              t1=t1, t2=5, c1=channel, c2=channel, l1=size_patch, l2=size_patch)
        else:
            input = rearrange(input, 'b (t1 c1) (l1 t2) -> b (t1 t2) (c1 l1)', t1=t1, t2=10, c1=16, l1=size_patch)
        return input

    def plot_images(self, indices, show: bool = True,
                    save: str = None,
                    fig_size=(5.5, 5),
                    vqvae: VQVAE = None):
        if self.resolution.get() == 0:
            rows = 10
        else:
            rows = 5
        images = vqvae.decode(indices)
        fig = plt.figure(figsize=fig_size)
        out: torch.Tensor = make_grid(images + 0.5, nrow=rows, padding=10)
        out_np: np.array = K.tensor_to_image(out)
        plt.imshow(out_np)
        plt.axis('off')
        if show:
            plt.show(block=True)
        if save is not None:
            plt.savefig(save)
        plt.close()
        return fig

    @staticmethod
    def plot_spectrogram(indices, show=True, save: str = None, vqvae: SpeechVQVAE = None, fig_size=(4.5, 3.5)):
        spec = vqvae.decode(indices)
        spec = np.sqrt(torch.transpose(spec.squeeze(1), 0, 1).cpu().detach().numpy())
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=fig_size)
        win_length = int(64e-3 * 16000)
        hop = int(0.625 / 2 * win_length)
        img = display.specshow(librosa.amplitude_to_db(spec, ref=np.max), sr=16000, y_axis='linear',
                               x_axis='time', hop_length=hop, win_length=win_length)
        plt.xlabel('Time (s)', fontsize=15)
        plt.ylabel('Frequency (Hz)', fontsize=15)
        # fig.colorbar(img, ax=ax, format="%+2.0f dB")
        if save is not None:
            plt.savefig(save)

        if show:
            plt.show(block=True)
        else:
            plt.close()
        return fig

    @staticmethod
    def save_animation(indices, vqvae: VQVAE, video_name: str = None, fig_size=(2, 2)):
        images = vqvae.decode(indices)
        images = images.cpu().detach().numpy()
        fps = 25
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes()
        plt.axis('off')
        grp = ax.imshow(np.transpose(images[0], (1, 2, 0)) + 0.5, 'gray')

        def update(frame_number):
            image = images[frame_number]
            grp.set_array(np.transpose(image, (1, 2, 0)) + 0.5)
            return grp,

        anim = FuncAnimation(fig, update, frames=images.shape[0], interval=1000 / fps, repeat=False)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=1800, metadata=dict(artist='VQ-MAE-AV'))
        if video_name is not None:
            anim.save(video_name, writer=writer)
        plt.show(block=True)

    def save_wav(self, indices, save: str = None, vqvae: SpeechVQVAE = None, phase=None, play: bool = False):
        audio = vqvae.decode(indices)
        audio = np.sqrt(torch.transpose(audio.squeeze(1), 0, 1).cpu().detach().numpy())
        if phase is not None:
            signal = self.istft(audio, phase)
        else:
            signal = self.griffin_lim(audio)
        if save is not None:
            write(save, 16000, signal)
        if play:
            sd.play(signal, samplerate=16000)

    @staticmethod
    def istft(stft_matrix, phase):
        win_length = int(64e-3 * 16000)
        hop = int(0.625 / 2 * win_length)
        convstft = stft_matrix * (np.cos(phase) + 1j * np.sin(phase))
        signal = librosa.istft(convstft, hop_length=hop, win_length=win_length)
        return signal

    @staticmethod
    def griffin_lim(spec):
        win_length = int(64e-3 * 16000)
        hop = int(0.625 / 2 * win_length)
        signal = librosa.griffinlim(spec, hop_length=hop, win_length=win_length, n_iter=500)
        return signal
