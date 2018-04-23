import vlc
import os
import cPickle as pickle
import time
from os.path import dirname, abspath, join


def main():
    preloaded_audio_dir = join(dirname(dirname(abspath(__file__))), 'databases/audio_vocabulary') + '/'
    # Maximum directory depth is 2
    sub_dirs = [preloaded_audio_dir + name + '/' for name in os.listdir(preloaded_audio_dir)]
    audio_duration_dict = {}
    def process_wav_dir(dir):
        audio_paths = [dir + name for name in os.listdir(dir)]
        for wav in audio_paths:
            audio_duration_dict[wav] = {}.fromkeys(['dur', 'int'])
            p = vlc.MediaPlayer(wav)
            p.play()
            audio_duration_dict[wav]['dur'] = p.get_length()
            # audio_duration_dict[wav]['int'] = get_intermission(wav)

    for dir in sub_dirs:
        # Check whether the dir contents are sub-dirs. Only can handle all subdirs or all dirs
        sub_dirs = [dir + name for name in os.listdir(dir)]
        if os.path.isdir(sub_dirs[0]): # Just check the first one
            sub_dirs = [dir + name + '/' for name in os.listdir(dir)]
            # Contents are subdirs. Each one can be extracted for durations
            for sub_dir in sub_dirs:
                process_wav_dir(sub_dir)
        else:
            process_wav_dir(dir)

    # Save result to dict
    save_pkl_path = join(dirname(dirname(abspath(__file__))), 'databases/oldaudio_duration_dict.pkl')
    pickle.dump(audio_duration_dict, open(save_pkl_path, 'wb'))

if __name__ == '__main__':
    main()
#  -----------------------------------------------------------------------------------------------
#  -----------------------------------------------------------------------------------------------
#  -----------------------------------------------------------------------------------------------
