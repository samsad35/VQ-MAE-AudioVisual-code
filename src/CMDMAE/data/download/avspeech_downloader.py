import sys
import os
from multiprocessing.pool import ThreadPool
from yt_dlp import YoutubeDL
import ffmpeg


class VidInfo:
    def __init__(self, yt_id, start_time, end_time, outdir):
        self.yt_id = yt_id
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.out_filename = os.path.join(outdir, yt_id + '_' + start_time + '_' + end_time + '.mp4')


def download(vidinfo):

    yt_base_url = 'https://www.youtube.com/watch?v='

    yt_url = yt_base_url+vidinfo.yt_id
    print(yt_url)

    ydl_opts = {
        'format': '22/18',
        'quiet': True,
        'ignoreerrors': True,
        'no_warnings': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            download_url = ydl.extract_info(url=yt_url, download=False)['url']
    except:
        return_msg = '{}, ERROR (youtube)!'.format(vidinfo.yt_id)
        return return_msg
    try:
        print(vidinfo.out_filename)
        (
            ffmpeg
                .input(download_url, ss=vidinfo.start_time, to=vidinfo.end_time)
                .output(vidinfo.out_filename, format='mp4', r=25, vcodec='libx264',
                        crf=18, preset='veryfast', pix_fmt='yuv420p', acodec='aac', audio_bitrate=128000,
                        strict='experimental')
                .global_args('-y')
                .global_args('-loglevel', 'error')
                .run()

        )
    except:
        return_msg = '{}, ERROR (ffmpeg)!'.format(vidinfo.yt_id)
        return return_msg

    return '{}, DONE!'.format(vidinfo.yt_id)


if __name__ == '__main__':

    split = "tmp"
    csv_file = r'D:\These\data\Audio-Visual\AVSpeech\avspeech_{}.csv'.format(split)
    out_dir = r"D:\These\data\Audio-Visual\AVSpeech\AVSpeech_train"

    os.makedirs(out_dir, exist_ok=True)

    with open(csv_file, 'r') as f:
        lines = f.readlines()
        print(lines)
        lines = [x.split(',') for x in lines]
        vidinfos = [VidInfo(x[0], x[1], x[2], out_dir) for x in lines]

    bad_files = open('bad_files_{}.txt'.format(split), 'w')
    results = ThreadPool(5).imap_unordered(download, vidinfos)
    cnt = 0
    for r in results:
        cnt += 1

