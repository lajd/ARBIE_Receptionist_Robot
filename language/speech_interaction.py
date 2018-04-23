import speech_recognition as sr
import vlc
import os
import time
import datetime
from numpy import arange
import cPickle as pickle
from numpy import argmin, argmax
from mutagen.mp3 import MP3
from os.path import dirname, abspath
import threading
from threading import Thread
from queue import Queue
import random

class microphone_manager(threading.Thread):
    def __init__(self, recognizer, microphone, durations_dict, speech_synthesizer):
        threading.Thread.__init__(self)
        self.recognizer = recognizer
        self.microphone = microphone
        self.default_dur = 0.6
        self.durations_dict = durations_dict
        self.commands = Queue()
        self.output = Queue()
        self.speech = speech_synthesizer
        self.ambient_noise = Queue()
        self.mod_counter = 5
        self.played_audio = Queue()
        self.is_adjusting = Queue()

        self.ambient_noise_monitor = []
        self.ambient_noise_monitor_buffer_size = 45
        self.timeout = 10
        self.phrase_timeout = 10

    def run(self):
        it = 0
        with self.microphone as source:
            while True:
                command = self.commands.get(True)
                if command['Command'] == 'Start':
                    wav_path = command['Wav_path']
                    if type(wav_path) == tuple or type(wav_path) == list:
                        dur = 0
                        for wp in wav_path:
                            dur += self.durations_dict[wp]['dur']
                    else:
                        dur = (self.durations_dict[wav_path]['dur'] - 0.6) if wav_path != None else self.default_dur
                    if dur > 0.5:
                        if self.played_audio.qsize() > 0:
                            pass
                        else:
                            self.is_adjusting.put(True)
                            self.recognizer.adjust_for_ambient_noise(self.microphone, dur)
                            self.is_adjusting.get()

                elif command['Command'] == "Listen":
                    print "Executing listen sequence"
                    # Because timeout doesn't always work
                    method_before_speech_analysis = command["method_before_speech_analysis"]
                    args = command["args"]

                    t1 = time.time()
                    self.speech.play_audio(os.path.join( self.speech.audio_vocab_dir, 'sounds/better_beep.mp3'), intermission=False)
                    print('----------------')
                    print('ready to listen!')
                    print('----------------')
                    t2 = time.time()
                    print 'total time is: {}'.format(t2-t1)
                    audio = self.recognizer.listen(source=source, timeout=10, phrase_time_limit=self.phrase_timeout)
                    print 'detector turned off in:', (time.time() - t1)
                    # Say something prior to speech analysis, if necessary
                    self.speech.play_random(os.path.join( self.speech.audio_vocab_dir, 'rand_lemme_think'))
                    if method_before_speech_analysis: method_before_speech_analysis(args)

                    # Try to launch a thread to adjust ambient noise
                    Thread(target=self.recognizer.adjust_for_ambient_noise, args=(self.microphone, 0.5)).start()

                    try:
                        print "Updated ambient energy is:", self.recognizer.energy_threshold
                        stt = self.recognizer.recognize_google_cloud(audio,
                                                                     credentials_json=self.speech.GOOGLE_CLOUD_SPEECH_CREDENTIALS)
                        print 'Response is: {0}'.format(stt)
                        self.output.put({"Output":stt})
                    except sr.WaitTimeoutError as e:
                        print("Timeout; {0}".format(e))
                        self.output.put({"Output": None})
                    except sr.UnknownValueError as e:
                        print("Google Speech Recognition could not understand audio; {0}".format(e))
                        self.output.put({"Output": None})
                    except sr.RequestError as e:
                        print("Could not request results from Google Speech Recognition service; {0}".format(e))
                        self.output.put({"Output": None})

                elif command['Command'] == "Pause":
                    continue

                while self.commands.qsize() > 0:
                    self.commands.get()
                self.ambient_noise.put({'energy':self.recognizer.energy_threshold})
                if it % self.mod_counter == 0:
                    print "Updated ambient energy is:", self.recognizer.energy_threshold
                it += 1

                self.ambient_noise_monitor.append(self.recognizer.energy_threshold)
                if len(self.ambient_noise_monitor) > self.ambient_noise_monitor_buffer_size:
                    self.ambient_noise_monitor = self.ambient_noise_monitor[-self.ambient_noise_monitor_buffer_size:]

class Speech(object):
    def __init__(self, ambient_noise_duration=0.5):
        self.main_dir = dirname(dirname(abspath(__file__)))
        self.audio_vocab_dir = os.path.join(self.main_dir, 'database/databases/audio_vocabulary/')
        self.db_initializer_dir = os.path.join(self.main_dir, 'database/databases/')
        self.ambient_noise_duration = ambient_noise_duration  # seconds to adjust for ambient noise

        self.recognizer = sr.Recognizer()
        # self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_ratio = 1.65 # Speaker must be twice as loud as ambient
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = 10

        self.microphone = sr.Microphone(self.check_device_microphones())

        # Reinitialize audio dict incase more files have been added.
        self.initialize_audio_dict()
        self.audio_durations_dict = pickle.load(open(os.path.join(self.db_initializer_dir, 'audio_duration_dict.pkl')))

        # Place JSON PRIVATE KEY here (not typically required)
        self.GOOGLE_CLOUD_SPEECH_CREDENTIALS = None

        self.microphone_mgr = microphone_manager(self.recognizer, self.microphone, self.audio_durations_dict, self)
        self.microphone_mgr.start()

        self.rand_wait = []

    def listen(self, method_before_speech_analysis=None, args=None):
        print "Listening with google speech"
        self.microphone_mgr.commands.put({"Command": "Listen", "method_before_speech_analysis":method_before_speech_analysis, "args":args})
        while True:
            outp = self.microphone_mgr.output.get(True)
            if outp != None:
                return outp['Output']

    @staticmethod
    def check_device_microphones():
        import pyaudio
        p = pyaudio.PyAudio()
        correct_idx = None
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            print((i, dev['name'], dev['maxInputChannels']))
            if 'micArray' in dev['name']:
                correct_idx = i
        return correct_idx

    def play_audio(self, wav_path, intermission=True):
        """
        PLays an audio file from its path
        :param wav_path: (str) path to audio
        :return: (audio) Plays audio file
        """

        def get_default_audio_intermission(wav):
            # TODO: Refactor this out into an 'intermission' dict
            # par_dir = self.parent_dir + 'audio_vocabulary/'
            if '/booking_confirmation_predate' in wav:
                return 0.5
            elif '/very_nice_to_meet_you.mp3' in wav:
                return 0.4
            elif '/hi.mp3' in wav:
                return 0.2
            elif '/months/' in wav or '/times/' in wav or '/joiners/' in wav:
                return 0.3
            elif 'numbers' in wav:
                return 0.3
            elif '/days/' in wav:
                return 0.1
            elif '/names/' in wav:
                return 1.2
            else:
                return 0.6

        def _play(wav_path, intermission=True):

            p = vlc.MediaPlayer(wav_path)
            p.play()
            # Pause for audio duration + default_audio_intermission
            default_audio_intermission = get_default_audio_intermission(wav_path)

            try:
                if intermission:
                    time.sleep(float(self.audio_durations_dict[wav_path]['dur']) + default_audio_intermission)
                    print(wav_path.split('/')[-1], (float(self.audio_durations_dict[wav_path]['dur'])) + default_audio_intermission)
                else:
                    _int = -0.2
                    time.sleep(float(self.audio_durations_dict[wav_path]['dur']) + _int)
                    print(wav_path.split('/')[-1], (float(self.audio_durations_dict[wav_path]['dur'])) + _int)

            except KeyError as e:
                # Assume word is terminal and doesn't require pause
                print('error during intermission: ', e)

        while True:
            # Wait until done previous adjustment
            if self.microphone_mgr.is_adjusting.qsize() > 0:
                continue
            else:
                break

        self.microphone_mgr.played_audio.put(True)
        if type(wav_path) == tuple or type(wav_path) == list:
            # If input is a list, play each file in the list
            for pth in wav_path:
                _play(pth, intermission=intermission)
        else:
            _play(wav_path, intermission=intermission)

        self.microphone_mgr.played_audio.get()

    def initialize_audio_dict(self):
        # Maximum directory depth is 2
        sub_dirs = [self.audio_vocab_dir + name + '/' for name in os.listdir(self.audio_vocab_dir)]
        audio_duration_dict = {}

        def process_wav_dir(dir):
            audio_paths = [dir + name for name in os.listdir(dir)]
            for wav in audio_paths:
                audio_duration_dict[wav] = {}.fromkeys(['dur', 'int'])
                audio = MP3(wav)
                audio_duration_dict[wav]['dur'] = audio.info.length
                # audio_duration_dict[wav]['int'] = get_intermission(wav)

        for dir in sub_dirs:
            # Check whether the dir contents are sub-dirs. Only can handle all subdirs or all dirs
            sub_dirs = [dir + name for name in os.listdir(dir)]
            if os.path.isdir(sub_dirs[0]):  # Just check the first one
                sub_dirs = [dir + name + '/' for name in os.listdir(dir)]
                # Contents are subdirs. Each one can be extracted for durations
                for sub_dir in sub_dirs:
                    process_wav_dir(sub_dir)
            else:
                process_wav_dir(dir)

        # Write the audio dur dict to path
        pickle.dump(audio_duration_dict, open(os.path.join(self.db_initializer_dir, 'audio_duration_dict.pkl'), 'wb'))
        print('Saved new audio duration dictionary')

    def update_audio_dur_dict(self, wav, intermission=None):
        self.audio_durations_dict[wav] = {}.fromkeys(['dur', 'int'])
        audio = MP3(wav)
        self.audio_durations_dict[wav]['dur'] = audio.info.length
        if intermission:
            self.audio_durations_dict[wav]['dur'] = intermission

    def play_random(self, dir_pth):
        audio_files = [os.path.join(dir_pth, x) for x in os.listdir(dir_pth)]
        rand_idx = random.randint(0, len(audio_files)-1)
        self.play_audio(audio_files[rand_idx])


class Translator():
    """
    This class instantiates and gives access to encoding/decoding dictionaries for speech parsing.

    Currently only the date/time is being encoded/decoded.

    The information considered here so far include:

    Month, Day, Time
    """

    def __init__(self):
        self.main_dir =  dirname(dirname(abspath(__file__)))

        self.date_time = datetime.datetime.now()
        self.current_weekday = time.strftime("%c").split(' ')[0].lower()
        self.current_year = self.date_time.year
        self.current_month = self.get_month().lower()
        self.current_day = self.date_time.day
        self.current_time = self.convert_time_hrs_pst_midnt(self.date_time.hour, self.date_time.minute,
                                                            self.date_time.second)
        self.audio_vocab_dir = os.path.join(self.main_dir, 'database/databases/audio_vocabulary/')

        # Load important information
        self.day_start = 8
        self.day_end = 18.5
        self.day_inc = 0.5
        self.default_audio_intermission = 0.5

        self.encode_day_dict = None
        self.decode_day_dict = None

        self.encode_month_dict = None
        self.decode_month_dict = None

        self.year_dict = None
        self.name_dict = None
        self.city_dict = None
        self.province_dict = None
        self.number_dict = None
        self.compass_headings = None

        self.encode_time_dict = None
        self.decode_time_dict = None
        self.affirmation_dict = self.initialize_affirmation_dict()

        # Initialize encode and decode dicts
        self.initialize_month_dict()
        self.initialize_day_dict()
        self.initialize_time_dict()
        self.initiate_number_dict()
        self.initialize_year_dict()
        self.initialize_name_dict()
        self.initialize_compass_headings()
        self.initialize_street_suffixes_dict()
        self.initialie_geography_dicts()
        self.initialize_affirmation_dict()

    def get_date_speech(self, month, day, year, time):
        """
        Gets the audio files corresponding to month, day and time
        :param month: (int)x
        :param day: (int)
        :param time: (float)
        :return: (list) wav files corresponding to month, day and time
        """
        # Speech thus far is .. "on"..
        month_wav = self.decode_month_dict[month]
        # .. [Jan]
        day_wav = self.decode_day_dict[day]
        # .. [5th]
        # say --> (at)
        # .. [at]
        say_wav = os.path.join(self.audio_vocab_dir, 'dates_and_times/joiners/at.mp3')
        # Say time
        # .. [5]

        # year
        if str(int(year)) == '2018':
            year_wav = self.audio_vocab_dir + 'new_profile/enhanced/2018.mp3'
        elif str(int(year)) == '2019':
            year_wav = self.audio_vocab_dir + 'new_profile/enhanced/2019.mp3'
        elif str(int(year)) == '2020':
            year_wav = self.audio_vocab_dir + 'new_profile/enhanced/2020.mp3'
        else:
            year_wav = self.audio_vocab_dir + 'new_profile/enhanced/2018.mp3'

        time_wav = self.decode_time_dict[time]

        # Get list of each of the waves
        say_list = [month_wav, day_wav, year_wav, say_wav, time_wav]

        return say_list

    def get_address(self, user_str, return_clues=False):
        # 84 holmes avenue Guelph Ontario
        # 742 manitou drive

        number = []
        street = []
        city = []
        province = []

        curr_idx = 0
        tokens_list = user_str.lower().split()
        for tok in tokens_list:
            if tok in self.number_dict:
                number.append(tok)
            elif tok in self.street_suffixes:
                street.append(tokens_list[curr_idx - 1] + ' ' + tok)
            elif tok in self.compass_headings:
                if len(street) > 0:
                    street[-1] += ' ' + tok
                else:
                    street.append(tokens_list[curr_idx - 1] + ' ' + tok)
            elif tok in self.city_dict:
                city.append(tok)
            elif tok in self.province_dict:
                province.append(tok)
            curr_idx += 1

        if return_clues:
            return number, street, city, province

        # Do some logic here
        if len(number) == 1 and len(street) == 1 and len(city) == 1 and len(province) == 1:
            return number[0], street[0], city[0], province[0]

        else:
            #  TODO: Make this better
            raise RuntimeError('Haven\'t accounted for this situation')

    def get_phone_number(self, usr_str):
        nums = [str(i) for i in range(10)]
        # Southern ontario phone numbers
        area_codes = [str(i) for i in [226, 289, 343, 416, 519, 613, 647, 705, 807, 905]]
        prefix = '1'
        prefix_area_codes = [prefix + i for i in area_codes]

        numeric_usr_str = ''.join([i for i in usr_str if i in nums])

        area_code = None
        number = None

        for ac in prefix_area_codes + area_codes:
            if ac in numeric_usr_str:
                area_code = ac
                number = numeric_usr_str.split(ac)[1][0:7]
                break
        print('area:{0}, phone:{1}'.format(area_code, number))
        return area_code, number


    def get_date_time(self, user_str, return_clues=False, allow_defaults=True):
        """
        Extracts the date from a str
        :param user_str: (str) (eg. from speech-to-text)
        :return: (list)  [Month, Day, Year, time]
        Month = (int) 0, 11..
        Day = (int) 1..31
        Year = (int) 2017
        time = (float) 8, 8.5, ..,  16
        """
        # Get rid of capitals.
        user_str = str.lower(str(user_str))

        # Split into toks
        toks = user_str.split(' ')

        # Initialize default values
        month = None
        day = None
        year = None
        time = None

        # Keep track of 'clues' so that we can check for inconsistencies and confirm with the user
        month_clues = []  # 0..12, if given
        day_clues = []  # 1..31, next tuesday
        time_clues = []  # 8:00, 2:30...
        year_clues = []  # The year, if given

        # Seperate tokens into potential clues
        for tok in toks:
            # Try to encode tokens based on encode dicts
            if tok in self.encode_month_dict:
                month_clues.append(self.encode_month_dict[tok])
            if tok in self.encode_day_dict:
                day_clues.append(self.encode_day_dict[tok])
            if tok in self.encode_time_dict:
                time_clues.append(self.encode_time_dict[tok])
            if tok in self.year_dict:
                year_clues.append(self.year_dict[tok])
            if tok == 'tomorrow':
                days_in_month = {0: 31, 1: 28, 2: 31, 3: 30, 4: 31, 5: 31, 6: 31, 7: 31, 8: 31, 9: 31, 10: 30, 11: 31}
                if (self.current_day + 1) <= days_in_month[self.encode_month_dict[self.current_month]]:
                    month_clues.append(self.encode_month_dict[self.current_month])
                    day_clues.append(self.current_day + 1)
                else:
                    month_clues.append(self.encode_month_dict[self.current_month] + 1)
                    day_clues.append(1)

        if allow_defaults:
            if len(year_clues) == 0:
                year_clues.append(self.current_year)

        # Date parse logic
        # Assume if a day of the week is given

        def get_day_given_weekday(weekday, time=None):
            """
            Get the day of the month given a weekday input
            :param weekday: (str) Eg. Tuesday
            :return: (int) assumed day of month
            """
            # Hardcode possible weekdays. Corresponds to output of google speech
            possible_weekdays = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
            # Hardcode how many days in each month
            # TODO: Implement auto leap-year finder method
            days_in_month = {0: 31, 1: 28, 2: 31, 3: 30, 4: 31, 5: 31, 6: 31, 7: 31, 8: 31, 9: 31, 10: 30, 11: 31}

            curr_weekday_idx = possible_weekdays.index(self.current_weekday)  # Current weekday
            suggested_weekday_idx = possible_weekdays.index(weekday)  # Weekday the user requested

            # If it's later in the week, then that's fine
            # If it's not, then we assume next week

            if suggested_weekday_idx > curr_weekday_idx:
                # is this week
                # Get number of days apart. Eg (wednesday --> Friday = 2)
                weekday_diffs = suggested_weekday_idx - curr_weekday_idx
                day = self.current_day + weekday_diffs
                if day > days_in_month[self.encode_month_dict[self.current_month]]:
                    day = weekday_diffs - (days_in_month[self.current_month] - self.current_day)
                return day
            else:
                # Is following week
                weekday_diffs = (6 - curr_weekday_idx) + (suggested_weekday_idx + 1)
                day = self.current_day + weekday_diffs
                if day > days_in_month[self.encode_month_dict[self.current_month]]:
                    print "current month:", self.current_month
                    print "days in month:", days_in_month

                    day = weekday_diffs - (days_in_month[self.encode_month_dict[self.current_month]] - self.current_day)
                return day

        def get_month_given_day(day_of_month):
            """
            Get the month given the day. Eg. the 5th might have passed and is assumed to be next month
            :param day_of_month: (int) eg. 5
            :return: (int) The assumed month.
            """
            if day_of_month > self.current_day:
                return self.encode_month_dict[self.current_month]
            else:
                return self.encode_month_dict[self.current_month] + 1 if self.encode_month_dict[self.current_month] + 1 <= 11 else 0

        # TODO
        def get_year_given_month(month_of_year):
            if month_of_year > self.current_month:
                return self.encode_month_dict[self.current_month]
            else:
                return self.encode_month_dict[self.current_month + 1 if self.current_month + 1 <= 11 else 0]

            pass

        if return_clues:
            return month_clues, day_clues, year_clues, time_clues

        else:
            # Do some basic logic of some edge cases
            # TODO: Add more edge cases
            if len(month_clues) == 1 and len(day_clues) == 1 and len(time_clues) == 1 and len(year_clues) == 1:
                m, d, y, t = month_clues[-1], day_clues[-1], year_clues[-1], time_clues[-1]

            # Next tuesday at 3:00
            elif len(month_clues) == 0 and len(day_clues) == 1 and len(time_clues) == 1:
                day = day_clues[0]
                if type(day) == int:
                    # Case where a numeric is given
                    month = get_month_given_day(day)
                elif type(day) == str:
                    # Case where input is "tuesday"
                    day = get_day_given_weekday(day)
                    month = get_month_given_day(day)
                    print day
                return month, day, year_clues[-1], time_clues[0]

            # TODO: Handle all the cases and handle when value is missing --> Ask questions to user
            else:
                m, d, y, t = month, day, year, time

            return m, d, y, t


    def initialize_month_dict(self):
        """
        Encode/decode months --> ints (eg. January = 0, December = 11) and vice versa
        :return: (n/a) updated self.{encode/decode}_month_dict
        """

        prepend_dir = 'dates_and_times/months/'

        # Keys corresponding to output from google speech
        encode_month_key = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                            'october', 'november', 'december']

        # Corresponding audio files for decoding
        decode_month_value = [self.audio_vocab_dir + prepend_dir + i + '.mp3' for i in encode_month_key]

        # Get the encoding/decoding
        encode_month_value = decode_month_key = range(12)

        encode_month_dict = {}.fromkeys(encode_month_key)
        decode_month_dict = {}.fromkeys(encode_month_key)

        # Populate encoding dictionary
        for i in range(len(encode_month_dict)):
            encode_month_dict[encode_month_key[i]] = encode_month_value[i]

        # Populate decoding dictionary
        for i in range(len(decode_month_dict)):
            decode_month_dict[decode_month_key[i]] = decode_month_value[i]

        self.encode_month_dict = encode_month_dict
        self.decode_month_dict = decode_month_dict

    def initialize_day_dict(self):
        """
        Encode/decode days --> ints (eg. 5th = 5, 15th = 15) and vice versa
        System has multiple encoding keys for robustness.
        Also accepts days of the week (eg Tuesday) as valid tokens, but doesn't assign meaningful value
        :return: (n/a) updated self.{encode/decode}_day_dict
        """

        prepend_dir = 'dates_and_times/days/'

        # Sometimes days are translated as words
        enc_keyset_1 = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth',
                        'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
                        'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'twenty first',
                        'twenty second', 'twenty third', 'twenty fourth', 'twenty fifth', 'twenty sixth',
                        'twenty seventh', 'twenty eighth', 'twenty ninth', 'thirtieth', 'thirty first']

        # Google speech typically translates as int
        enc_keyset_2 = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th',
                        '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '22nd', '23rd',
                        '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st']

        # Sometimes day information will be of the form 'next tuesday'.
        enc_keyset_3 = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
        # We encode in the following form for consistency with the time.get_current_time method
        enc_keyset_3_values = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']

        # Encoding values and decoding keys
        encoding_day_value = decoding_day_key = range(1, 32)

        # Value for decoding is just audio files corresponding to days
        decode_days_value = [self.audio_vocab_dir + prepend_dir + i.replace(' ', '-') + '.mp3' for i in
                             enc_keyset_1]

        encode_days = {}.fromkeys(enc_keyset_2 + enc_keyset_1)
        decode_days = {}.fromkeys(encoding_day_value)

        # Add keysets 1 and 2 to encoding dict
        for i in xrange(len(enc_keyset_2)):
            encode_days[enc_keyset_1[i]] = encoding_day_value[i]
            encode_days[enc_keyset_2[i]] = encoding_day_value[i]

        # Add keyset 3 to encoding dict
        for i in xrange(len(enc_keyset_3_values)):
            encode_days[enc_keyset_3[i]] = enc_keyset_3_values[i]

        # Add decoding
        for i in xrange(len(decoding_day_key)):
            decode_days[decoding_day_key[i]] = decode_days_value[i]

        self.decode_day_dict = decode_days
        self.encode_day_dict = encode_days

    def initialize_time_dict(self):
        """
        Encode/Decode times to int
        Times are coming frmo google speech API output
        :return: (n/a) Updated self.time_dict
        """

        # Assumes the day increment to be 30 min
        # Typical google speech API output. Also mimicks naming convention of preloaded_audio time files
        enc_keyset_1 = time_file_names = ['8', '830', '9', '930', '10', '1030', '11', '1130', '12', '1230', '1',
                                          '130', '2', '230', '3', '330', '4', '430', '5', '530', '6']

        # Typical google speech API output
        enc_keyset_2 = ['8:00', '8:30', '9:00', '9:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30',
                        '1:00', '1:30', '2:00', '2:30', '3:00', '3:30', '4:00', '4:30', '5:00', '5:30', '6:00']

        # Encoding values and decoding keys
        encode_values = decode_keys = [8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5,
                                       16, 16.5, 17, 17.5, 18]

        encode_time_dict = {}.fromkeys(enc_keyset_1 + enc_keyset_2)
        decode_time_dict = {}.fromkeys(decode_keys)

        # Update the encoding dictionary
        for i in range(len(enc_keyset_1)):
            encode_time_dict[enc_keyset_1[i]] = encode_values[i]
            encode_time_dict[enc_keyset_2[i]] = encode_values[i]

        # For decoding, need to check map float back to wav file.
        prepend_dir = 'dates_and_times/times/'  # Directory where day audio files are located

        for i in range(len(decode_keys)):
            decode_time_dict[decode_keys[i]] = []  # List since needs more than 1 value

            time_file_name = time_file_names[i]
            bse_hour = int(decode_keys[i])

            if '30' in time_file_name:
                # Time falls on a half hour increment

                # TODO: Don't hard code the day incs
                decode_time_dict[decode_keys[i]].append(
                    (self.audio_vocab_dir + prepend_dir + time_file_name[0:-2] + '.mp3',
                     self.audio_vocab_dir + prepend_dir + '30.mp3'))

                # Check how to finish off saying the time  -- E.g. 'o'clock am' vs 'pm'
                if float(bse_hour) >= 12:
                    decode_time_dict[decode_keys[i]].append(self.audio_vocab_dir + 'dates_and_times/times/pm.mp3')
                else:
                    decode_time_dict[decode_keys[i]].append(self.audio_vocab_dir + 'dates_and_times/times/am.mp3')
            else:
                # Time falls on the hour
                decode_time_dict[decode_keys[i]].append(self.audio_vocab_dir + prepend_dir + time_file_name + '.mp3')

                if float(bse_hour) >= 12:
                    decode_time_dict[decode_keys[i]].append(
                        self.audio_vocab_dir + 'dates_and_times/times/oclock_pm.mp3')
                else:
                    decode_time_dict[decode_keys[i]].append(
                        self.audio_vocab_dir + 'dates_and_times/times/oclock_am.mp3')

        # Add some custom logic
        #######################
        encode_time_dict['noon'] = 12

        self.encode_time_dict = encode_time_dict
        self.decode_time_dict = decode_time_dict

    def initialize_year_dict(self):
        years = range(1900, 2050)
        yr_dict = {}
        for i in years:
            yr_dict[str(i)] = float(i)
        self.year_dict = yr_dict

    def initialize_name_dict(self):
        name_dict = pickle.load(
            open(os.path.join(self.main_dir, 'database/databases/parse_dicts/names.pkl'), 'r'))
        self.name_dict = {}
        for name in name_dict:
            self.name_dict[(name.lower())] = None


    def initialie_geography_dicts(self):
        with open(
                os.path.join(self.main_dir, 'database/databases/parse_dicts/raw/list_of_cities_of_canada-1634j.csv'),
                'r') as f:
            content = f.readlines()
            content = [i.strip().split(',') for i in content]

            provinces = ['Nova Scotia', 'British Columbia', 'Newfoundland and Labrador', 'Saskatchewan',
                         'Prince Edward Island',
                         'Ontario', 'Quebec', 'Alberta', 'Manitoba', 'Northwest Territories', 'New Brunswick',
                         'Nunavut', 'Yukon']
            cities = list(set([i[2] for i in content[1:]]))

            lower_provinces = [i.lower() for i in provinces]
            lower_cities = [i.lower() for i in cities]
            self.province_dict = dict.fromkeys(lower_provinces)
            self.city_dict = dict.fromkeys(lower_cities)

    def initiate_number_dict(self):
        range_vals = [str(i) for i in range(10000)]
        self.number_dict = {}.fromkeys(range_vals)

    def initialize_street_suffixes_dict(self):
        full_suffixes = []
        abbreviated_suffixes = []
        # "Alley" Aly Ending
        with open(os.path.join(self.main_dir, 'database/databases/parse_dicts/raw/street_suffix'), 'r') as f:
            content = [i.strip().split(' ') for i in f.readlines()]
            for line in content:
                full_suffixes.append(line[0].strip('\"'))
                abbreviated_suffixes.append(line[1])

        lower_keys = [i.lower() for i in (full_suffixes + abbreviated_suffixes)]
        self.street_suffixes = {}.fromkeys(lower_keys)

    def initialize_compass_headings(self):
        headings_dict = {'north': 'North',
                         'east': 'East',
                         'south': 'South',
                         'west': 'West'
                         }
        self.compass_headings = headings_dict

    def get_month(self):
        month_strs = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                      'October', 'November', 'December']
        return month_strs[self.date_time.month - 1]

    def snap_time(self, time):
        """
        Function snaps time to closest day_inc : eg. 11:29 -> 11:30
        :param time: (float)
        :return: (float) snapped time
        """
        time_ranges = list(arange(self.day_start, self.day_end + self.day_inc, self.day_inc))

        if time not in time_ranges:
            time = time_ranges[argmin([abs(time - i) for i in time_ranges])]
        return time

    def convert_time_hrs_pst_midnt(self, hrs, min, sec):
        """
        Converts time in hours, minutes and seconds to time since midnight in hours
        :param hrs: (float)
        :param min: (float)
        :param sec: (float)
        :return: (float)
        """
        return hrs + float(min) / 60 + float(sec) / 3600

    def initialize_affirmation_dict(self):
        affirmative = {}

        yes_words = ['yes', 'right', 'true', 'correct', 'yup', 'yeah', 'ya', 'ye', 'sure']
        no_words = ['no', 'wrong', 'false', 'incorrect', 'nope', 'not', 'nah']

        for i in yes_words:
            affirmative[i] = True
        for i in no_words:
            affirmative[i] = False

        return affirmative

    def affirmation(self, resp):
        affirm_lst = []
        for i in resp.split(" "):
            try:
                affirm_lst.append(self.affirmation_dict[i])
            except KeyError:
                pass
        if affirm_lst.count(True) > 0 and (affirm_lst.count(True) > affirm_lst.count(False)):
            return True
        else:
            return False

    def reinitialize_name_dict(self):
        with open(os.path.join(self.main_dir, 'database/databases/parse_dicts/raw/names.txt'), 'r') as f:
            content = f.readlines()
            content = [line.strip() for line in content]
            names_dict = {}.fromkeys(content)
            with open(os.path.join(self.main_dir, 'database/databases/parse_dicts/raw/names.pkl'), 'wb') as f1:
                pickle.dump(names_dict, f1)




