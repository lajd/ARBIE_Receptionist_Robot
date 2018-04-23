from language.speech_interaction import Speech, Translator
from tinydb import TinyDB, Query
from tinydb import Query
from tinydb.operations import set as dbset
import cPickle as pickle
from copy import deepcopy
from vision.facenet_recognition import FacialRecognition
import boto3
from threading import Thread
import os
from os.path import abspath, dirname
from os.path import join
from queue import Queue
import time
import datetime
from scipy.stats import zscore
from language.snowboy.snowboy_threaded import ThreadedDetector


class snowboy_detection_manager(object):
    def __init__(self, speech_synthesizer, timeout=10, global_sensitivity=0.3):
        self.ARBIE_dir = dirname(abspath(__file__)) + '/'
        # The sentitivity of offline word detection for each word
        self.sens_dict = {'yes':0.40,
                     'no':0.4,
                     'book':0.45,
                     'when':0.5,
                     'appointment':0.5,
                     'check':0.4,
                     'schedule':0.4,
                      'delete':0.1,
                        'cancel':0.1,
                          'unused':0.1}

        self.decoder_pth = join(self.ARBIE_dir, 'language/snowboy/decoder_models')
        self.timeout = timeout
        self.hotword_manager = {}
        self.global_sensitivity = global_sensitivity
        self.initialize()
        self.primed = False
        self.speech = speech_synthesizer

    def _size(self):
        count = 0
        for word in self.hotword_manager:
            count += len(self.hotword_manager[word])
        return count

    def listen_for_fires(self, word_dict=None):
        """
        This is only called once per recognition event

        get_fired_words is then called until results are obtained
        :return:
        """
        print 'Beginning listening sequence'
        self.speech.play_audio(os.path.join(self.speech.audio_vocab_dir, 'sounds/eh.mp3'))

        for word in self.hotword_manager.keys():
            if word_dict:
                if word not in word_dict:
                    continue
            for thread in self.hotword_manager[word]:
                t = self.hotword_manager[word][thread]
                t.start_recog()

        print 'Finished listening sequence'

    def is_alive_running(self):
        isalive = True
        isrunning = False
        for word in self.hotword_manager:
            for thread in self.hotword_manager[word]:
                t = self.hotword_manager[word][thread]
                if t.isAlive() == False:
                    isalive = False
                if t.is_running() == True:
                    isrunning = True
        return isalive, isrunning

    def get_fired_words(self, z_thresh=1.2):
        word_counts_dict = self._collect_responses()
        # Gaussian probability of word occurence
        word_zscores = zscore(word_counts_dict.values())
        key_words = {}
        for k, v in [(key, val) for key, val in zip(word_counts_dict.keys(), word_zscores) if val > z_thresh]:
            key_words[k] = v
        print key_words
        return key_words

    def initialize(self):
        print 'Started initializing hotword threads'
        category_list = [join(self.decoder_pth, i) for i in os.listdir(self.decoder_pth)]
        for cat_pth in category_list:
            for word in os.listdir(cat_pth):  # word, path
                self.hotword_manager[word] = {}
                for model in os.listdir(join(cat_pth, word)):
                    model_pth = join(cat_pth, word, model)
                    self.hotword_manager[word][model] = ThreadedDetector([model_pth], sensitivity=  self.sens_dict[word]) #self.global_sensitivity)
                    self.hotword_manager[word][model].start()  # Start the thread
        print 'Ended initializing hotword threads'

    def pause_firing(self, word_dict):
        print 'Beginning pause sequence'

        def _clear(word_dict):
            for word in self.hotword_manager:
                if word_dict:
                    if word not in word_dict:
                        continue
                    else:
                        for model in self.hotword_manager[word]:
                            self.hotword_manager[word][model].pause_recog()
                            while True:
                                if self.hotword_manager[word][model].is_running():
                                    continue
                                else:
                                    break
                            assert self.hotword_manager[word][model].fired == 0
        _clear(word_dict)

        while True:
            if sum(self._collect_responses().values()) != 0:
                clear_keys = [k for k, v in self._collect_responses().items() if v != 0]
                _clear({}.fromkeys(clear_keys))
            else:
                break

        print 'Finished pause sequence'

    def _whos_running(self):
        outp = []
        for word in self.hotword_manager:
            for model in self.hotword_manager[word]:
                if self.hotword_manager[word][model].is_running():
                    outp.append([word, model])
        return outp

    def _collect_responses(self):
        words = {}
        for word in self.hotword_manager:
            words[word] = 0
            for model in self.hotword_manager[word]:
                words[word] += self.hotword_manager[word][model].fired
        return words

class ARBIE(object):
    def __init__(self, vision=False, ssh_to_pi=False, test=False, offline=100):

        self.speech = Speech()  # Communication
        self.translator = Translator()  # Encoding and decoding information

        if vision: self.vision = FacialRecognition()  # Vision

        self.ARBIE_dir = dirname(abspath(__file__)) + '/'
        self.audio_vocab_dir = join(self.ARBIE_dir, 'database/databases/audio_vocabulary/')
        self.db_query = Query()
        self.appointment_db = TinyDB(join(self.ARBIE_dir, 'database/databases/appointment_scheduling.json'))
        self.profile_db = TinyDB(join(self.ARBIE_dir, 'database/databases/client_profiles.json'))
        self.intermission_dict = pickle.load(open(join(self.ARBIE_dir, 'database/databases/audio_duration_dict.pkl'), 'r'))
        self.translator = Translator()
        self.polly = boto3.client('polly', region_name='us-west-2')

        # Declare initialiation
        self.speech.play_audio(join(self.audio_vocab_dir, 'personal/arbie_initialized.mp3'))

        self.ssh_to_pi = ssh_to_pi
        self.cancel_sequence_interrupt = False
        self.vision_recognition_mod_n = 2
        self.queue = Queue()
        self.snowboy_detector = snowboy_detection_manager(self.speech, timeout=10)

        self.offline = offline
        self.test = test

    def watch_and_listen(self, send_to_pi=True):

        # Watch for individuals. If recognize individuals, say hi and help them find the information they need
        # If you don't recognize the individual, ask them to register a new profile

        if self.test:
            pass
        else:
            Thread(target=self.vision.process_in_proximity, args=(self.speech, self.translator, self.new_user_profile, self.run_affirmation, send_to_pi, self.vision_recognition_mod_n, self.queue, self.polly)).start()

        while True:
            # Execute tasks, such as booking, etc.
            if self.queue.qsize() > 0:
                print 'starting general loop'

                next_queue_item = self.queue.get()
                if 'client' in next_queue_item:
                    client = next_queue_item['client']

                if 'appointment' in next_queue_item and next_queue_item['appointment'] == 'execute':
                    self.queue.put({'appointment': 'execute'})  # Put an item in the queue so recognition does not run
                    print 'FROM FACE DETECT executing general listen sequence'

                    def _execute_listen():
                        print 'executing general listen FROM FACE DETECT'
                        try:
                            self.general_listen_sequence(next_queue_item['client'])
                            return
                        except Exception as e:
                            # self.speech.microphone_mgr.commands.put({'Command':'Start', 'Wav_path':join(self.audio_vocab_dir, 'personal/how_can_help_you.mp3')})
                            self.speech.play_audio(join(self.audio_vocab_dir, 'personal/how_can_help_you.mp3'))
                            _execute_listen()
                    self.queue.get()  # Remove the item so that recognition can run

                    print 'exiting out of loop'

                elif 'check_for_appointment' in next_queue_item and next_queue_item['check_for_appointment'] == 'execute':
                    print 'checking for app'
                    self.queue.put({'check_for_appointment': 'execute', 'client':client})  # Put an item in the queue so recognition does not run

                    entries = self.lookup_persons_next_appointment(client=client)

                    if len(entries) > 0:
                        closest_entry = entries[0]
                        next_appt_epoch_time = closest_entry['Timestamp']

                        current_datetime = datetime.datetime.fromtimestamp(int(time.time()))
                        next_app_datetime = datetime.datetime.fromtimestamp(int(next_appt_epoch_time))

                        wavs = []

                        # Check if the date is today
                        if current_datetime.year == next_app_datetime.year and \
                            current_datetime.month == next_app_datetime.month and \
                            current_datetime.day == next_app_datetime.day:

                            wavs.append(join(self.audio_vocab_dir, 'appointment_management/your_next_app_not_for_another.mp3'))

                            def snap_min(x, base=10):
                                return int(base * round(float(x) / base))

                            minute_diff = str(snap_min((next_appt_epoch_time - time.time())/60))
                            save_pths = []
                            if int(minute_diff) > 60:
                                hours = int(minute_diff) // 60
                                save_hr_pth = self.audio_vocab_dir + 'numbers/' + str(hours) + '.mp3'
                                save_pths.append([save_hr_pth, hours])
                                minutes = int(minute_diff) - hours*60
                                save_min_pth = self.audio_vocab_dir + 'numbers/' + str(minutes) + '.mp3'
                                save_pths.append([save_min_pth, minutes])

                            else:
                                save_min_pth = self.audio_vocab_dir + 'numbers/' + str(minute_diff) + '.mp3'
                                save_pths.append([save_min_pth, minute_diff])

                            # Check if wav path already exists:
                            for pth, num in save_pths:

                                if pth in self.speech.audio_durations_dict:
                                    pass
                                else:
                                    # Create it otherwise
                                    spoken_text = self.polly.synthesize_speech(Text=str(num), OutputFormat='mp3',
                                                                               VoiceId='Joanna')
                                    with open(pth, 'wb') as f:
                                        f.write(spoken_text['AudioStream'].read())
                                    self.speech.update_audio_dur_dict(pth, intermission=0.4)

                            if len(save_pths) == 1:
                                wavs.append(save_min_pth)
                                wavs.append(join(self.audio_vocab_dir, 'appointment_management/minutes.mp3'))
                            else:
                                wavs.append(save_hr_pth)
                                wavs.append(join(self.audio_vocab_dir, 'appointment_management/hours.mp3'))
                                wavs.append(join(self.audio_vocab_dir, 'personal/and.mp3'))
                                wavs.append(save_min_pth)
                                wavs.append(join(self.audio_vocab_dir, 'appointment_management/minutes.mp3'))

                            wavs.append(join(self.audio_vocab_dir, 'appointment_management/help_anything_mean_time.mp3'))
                            wavs.append(join(self.audio_vocab_dir, 'personal/have_an_amazing_day.mp3'))

                        else:
                            wavs.append(join(self.audio_vocab_dir, 'appointment_management/dont_have_app_today_anything_i_can_do.mp3'))

                            wavs.append(join(self.audio_vocab_dir, 'personal/have_an_amazing_day.mp3'))

                        assert type(wavs[0:-1]) == list
                        self.speech.play_audio(wavs[0:-1])

                        wavs.append([join(self.audio_vocab_dir, 'personal/okay.mp3'), join(self.audio_vocab_dir,'personal/goodbye.mp3')])

                        def decline_fn():
                            wavs = []
                            wavs += [join(self.audio_vocab_dir, 'personal/okay.mp3'),
                                     join(self.audio_vocab_dir, 'personal/goodbye.mp3')]
                            self.speech.play_audio(wavs)

                        self._exit_affirmation(client=client, run_init=False, decline_fn=decline_fn)

                        print 'exiting out of loop hi->jemicy->no can do'
                    else:
                        print "No appointments found"

                        self.speech.play_audio(join(self.audio_vocab_dir, 'personal/no_appointments_like_to_book_one.mp3'))

                        def affirm_fn():
                            self.speech.play_audio(join(self.audio_vocab_dir, 'personal/amazing.mp3'))
                            self.book_new_appointment(client=client)

                        def decline_fn():
                            self.speech.play_audio(
                                join(self.audio_vocab_dir, 'personal/look_forward_to_seeing_you_again_soon.mp3'))
                            self.speech.play_audio(join(self.audio_vocab_dir, 'personal/goodbye.mp3'))
                            return 'EOC'

                        outp = self._exit_affirmation(run_init=False, affirmative_fn=affirm_fn, decline_fn=decline_fn,
                                                      affirmation_remark='arbitrary', client=client)
                    self.queue.get()  # Remove the item so that recognition can run

                elif 'new_user_profile' in next_queue_item and next_queue_item['new_user_profile'] == 'execute':
                    print ('executing new user profile')

                    t1 = time.time()

                    self.queue.put(
                        {'new_user_profile': 'execute'})  # Put an item in the queue so recognition does not run

                    if next_queue_item['questions'] == True:
                        self.new_user_profile(name=client)

                    while True:
                        if self.queue.qsize() > 1:
                            next_queue_item = self.queue.get()
                            if 'still_collecting_images' in next_queue_item:

                                if time.time() - t1 > 5:
                                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/almost_done.mp3'))

                                self.speech.play_audio(
                                    join(self.audio_vocab_dir, 'personal/i_remember_you.mp3'))
                                self.speech.play_audio(join(self.audio_vocab_dir, 'only_function_book_and_check.mp3'))
                                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/first_appointment.mp3'))

                                def affirm_fn():
                                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/amazing.mp3'))
                                    self.book_new_appointment(client=client)

                                def decline_fn():
                                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/look_forward_to_seeing_you_again_soon.mp3'))
                                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/goodbye.mp3'))
                                    return 'EOC'

                                outp = self._exit_affirmation(run_init=False, affirmative_fn=affirm_fn, decline_fn=decline_fn, affirmation_remark='arbitrary', client=client)
                                # Done processing, return to watch/listen
                                self.vision.interaction_end_time = time.time()
                                break

                    while self.queue.qsize() > 0:
                        self.queue.get()  # Remove the item so that recognition can run

                # Check if a global classifier should be retrained
                self.vision.scheduled_global_classifier()
                print 'ending loop'

    def general_listen_sequence(self,client=None, offline=None):
        if offline == None:
            offline = self.offline

        if type(offline) == float:
            noise = self.speech.microphone_mgr.ambient_noise.get()['energy']
            if noise < offline:
                offline = True
                print "Using offline due to low ambient energy"
            else:
                offline = False

        print 'Executing general listen sequence'

        def repeat_listen(word_dict=None, offline=False):
            """
            Ask 'how can I help you' again.
            """
            # Timeout
            if offline == True:
                self.snowboy_detector.pause_firing(word_dict)

                if len(self.snowboy_detector._whos_running()) > 0:
                    print self.snowboy_detector._whos_running()

            self.speech.play_audio(join(self.audio_vocab_dir, 'personal/didnt_understand.mp3'))
            time.sleep(1.5)
            self.speech.play_audio(join(self.audio_vocab_dir, 'personal/how_can_help_you.mp3'))

            # Check to make sure all threads have terminated
            if offline == True:
                alive, running = self.snowboy_detector.is_alive_running()
                while True:
                    if running:
                        continue
                    else:
                        break
            else:
                resp = self.speech.listen()
                return resp

        if offline == True:
            word_dict = {'book', 'when', 'appointment'}
            self.snowboy_detector.listen_for_fires(word_dict)
        else:
            resp = self.speech.listen()
            if resp == None:
                while resp == None:
                    resp = repeat_listen()
            fires = resp.lower().split(' ')

        start_time = time.time()
        min_time = 2

        iter = 0
        mod_count = 10

        while time.time() - start_time < self.snowboy_detector.timeout:
            if offline==True:
                fires = self.snowboy_detector.get_fired_words()

            if iter % mod_count != 0:
                continue

            if (time.time() - start_time) < min_time:
                # Skip if not min time yet, or else it sounds unnatural
                continue

            if 'book' in fires and 'when' in fires:
                if fires['book'] > fires['when']:
                    del fires['when']
                else:
                    del fires['book']

            if 'book' in fires and 'appointment' in fires:

                if offline==True:
                    self.snowboy_detector.pause_firing(word_dict)

                    if len(self.snowboy_detector._whos_running()) > 0:
                        print self.snowboy_detector._whos_running()

                time.sleep(2)

                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/okay_just_one_sec.mp3'))
                print 'Executing Appointment booking'
                self.book_new_appointment(client=client)
                print 'Finished executing appointment booking'
                eoc = self._exit_affirmation(run_init=True, client=client)
                "returning to calling function from general listen"
                return

            elif ('when' in fires and 'appointment' in fires) or ('check' in fires and 'schedule' in fires) :
                if offline==True:
                    self.snowboy_detector.pause_firing(word_dict)

                    if len(self.snowboy_detector._whos_running()) > 0:
                        print self.snowboy_detector._whos_running()

                time.sleep(2)

                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/let_me_look_that_up.mp3'))
                print 'executing checking appt'
                entries = self.lookup_persons_next_appointment(client=client)
                if len(entries) > 0:
                    closest_entry = entries[0]
                    next_appt_epoch_time = closest_entry['Timestamp']
                    current_datetime = datetime.datetime.fromtimestamp(int(time.time()))
                    next_app_datetime = datetime.datetime.fromtimestamp(int(next_appt_epoch_time))
                    wavs = []
                    # Check if the date is today
                    if current_datetime.year == next_app_datetime.year and \
                                    current_datetime.month == next_app_datetime.month and \
                                    current_datetime.day == next_app_datetime.day:
                        wavs.append(join(self.audio_vocab_dir, 'appointment_management/appointment_today.mp3'))

                    else:
                        wavs.append(join(self.audio_vocab_dir, 'appointment_management/your_next_app_on.mp3'))
                        wavs.append(self.translator.decode_month_dict[next_app_datetime.month - 1])
                        wavs.append(self.translator.decode_day_dict[next_app_datetime.day])
                        wavs.append(join(self.audio_vocab_dir, 'dates_and_times/joiners/at.mp3'))

                    hr = next_app_datetime.time().hour
                    thirty = True if abs(next_app_datetime.time().minute - 30) < 1e-5 else False
                    if thirty:
                        time_since_midnight = hr + 0.5
                    else:
                        time_since_midnight = hr

                    wavs += self.translator.decode_time_dict[time_since_midnight]

                    for wav in wavs:
                        self.speech.play_audio(wav)

                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/anything_else_can_help_with.mp3'))

                    def affirm_fn():
                        self.general_listen_sequence(client=client)

                    eoc = self._exit_affirmation(client=client, run_init=False, affirmative_fn=affirm_fn)
                    print eoc
                    "returning to calling function from general listen"
                    return

                else:

                    if offline==True:
                        self.snowboy_detector.pause_firing(word_dict)

                        if len(self.snowboy_detector._whos_running()) > 0:
                            print self.snowboy_detector._whos_running()

                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/no_appointments.mp3'))
                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/how_can_help_you.mp3'))
                    self.general_listen_sequence(client=client)
                    print "Continuing conversation"

            elif 'cancel' in fires:

                self.snowboy_detector.pause_firing(word_dict)

                if len(self.snowboy_detector._whos_running()) > 0:
                    print self.snowboy_detector._whos_running()

                time.sleep(1.5)

                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/thanks_for_chatting.mp3'))
                "returning to calling function from general listen"
                return

        if offline==True:
            repeat_listen(word_dict)
        else:
            repeat_listen()

    def is_truthy(self, wd_list):
        yes = {'yes', 'yep', 'sure', 'yeah', 'okay', 'yup', 'alright', 'absolutely'}
        no = {'no', 'nope', 'nay', 'neigh', 'nein', 'naw'}

        for wd in wd_list:
            if wd in yes:
                return True
            if wd in no:
                return False

        return None

    def run_affirmation(self, offline=None):
        """
        :param type: affirmation, ARBIE, appointment
        :return:
        """
        print "Running affirmation"

        if offline == None:
            offline = self.offline

        if type(offline) == float:
            noise = self.speech.microphone_mgr.ambient_noise.get()['energy']
            print "AMBIENT ENERGY IS: {}".format(noise)
            if noise < offline:
                offline = True
                print "USING OFFLINE MODE DUE TO LOW AMBIENT ENERGY"
            else:
                offline = False

        print 'OFFLINE IS: {} '.format(offline)
        if offline==True:
            start_time = time.time()
            word_dict = {'yes', 'no'}

            self.snowboy_detector.listen_for_fires(word_dict)

            while time.time() - start_time < self.snowboy_detector.timeout:
                fires = self.snowboy_detector.get_fired_words()
                outp = None
                if 'yes' in fires and 'no' in fires:
                    # say couldn't understand
                    if fires['yes'] > fires['no']:
                        self.snowboy_detector.pause_firing(word_dict)

                        if len(self.snowboy_detector._whos_running()) > 0:
                            print self.snowboy_detector._whos_running()

                        outp = True
                    else:
                        self.snowboy_detector.pause_firing(word_dict)

                        if len(self.snowboy_detector._whos_running()) > 0:
                            print self.snowboy_detector._whos_running()
                        outp = False
                    break
                elif self.is_truthy(fires) == True:
                    self.snowboy_detector.pause_firing(word_dict)

                    if len(self.snowboy_detector._whos_running()) > 0:
                        print self.snowboy_detector._whos_running()
                    outp=True
                    break
                elif self.is_truthy(fires) == False:
                    self.snowboy_detector.pause_firing(word_dict)

                    if len(self.snowboy_detector._whos_running()) > 0:
                        print self.snowboy_detector._whos_running()
                    outp=False
                    break

            self.snowboy_detector.pause_firing(word_dict)

            if len(self.snowboy_detector._whos_running()) > 0:
                print self.snowboy_detector._whos_running()
            return outp
        else:
            resp = self.speech.listen()
            if resp == None:
                return None
            print resp
            fires = resp.lower().split(' ')

            if self.is_truthy(fires) == False:
                return False
            elif self.is_truthy(fires) == True:
                return True
            else:
                return None

    def _exit_affirmation(self, client, run_init=False, init_remark=None, affirmative_fn=None, decline_fn=None, affirmation_remark=None, **kwargs):
        print "Running exit affirmation"

        def _run_affirmation(client, run_init=run_init, init_remark=None, affirmative_fn=None, decline_fn=None, affirmation_remark=None):
            if run_init:

                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/anything_else_can_help_with.mp3'))

            if init_remark:
                init_remark(**kwargs)

            affirmation = self.run_affirmation()
            if affirmation == False:
                if decline_fn:
                    decline_fn(**kwargs)
                else:
                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/have_an_amazing_day.mp3'))
                print "Finished Conversation"
                self.vision.interaction_end_time = time.time()
                return 'EOC'

            elif affirmation == True:
                if affirmation_remark == None:
                    # self.speech.microphone_mgr.commands.put({'Command':'Start', 'Wav_path':join(self.audio_vocab_dir, 'personal/how_can_help_you.mp3')})
                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/how_can_help_you.mp3'))
                if affirmative_fn:
                    affirmative_fn(**kwargs)
                    self.speech.play_audio(join(self.audio_vocab_dir, 'personal/anything_else_can_help_with.mp3'))
                    _run_affirmation(client=client, run_init=False, init_remark=None, affirmative_fn=None, decline_fn=None, affirmation_remark=None)
                    # return 'Completed request'
                else:
                    self.general_listen_sequence(client=client)
                    print "Continuing Conversation"
            else:

                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/didnt_understand.mp3'))
                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/anything_else_can_help_with.mp3'))
                _run_affirmation(client=client, run_init=False)
        res = _run_affirmation(client, run_init=run_init, init_remark=init_remark, affirmative_fn=affirmative_fn,
                               decline_fn=decline_fn, affirmation_remark=affirmation_remark)
        return res

    def book_new_appointment(self, play_init=True, client=None):
        """
        Begins listening after the beep. Then parses words and books an appointment on the date provided
        :return: Confirmation booking
        """

        print 'booking new appt'
        # Ask the user when they would like the appointment
        if play_init:

            self.speech.play_audio(join(self.audio_vocab_dir, 'appointment_management/when_would_appt.mp3'))

        # Start to listen for user speech
        user_speech = self.speech.listen(method_before_speech_analysis=self.speech.play_audio,
            args=join(self.audio_vocab_dir, 'appointment_management/okay_1_second_check_time_available.mp3'))

        print "Obtained requested booking date"

        month, day, year, time_of_day = self.translator.get_date_time(user_speech)

        def is_valid_id(m, d, y, t):

            if m != None and d != None and y != None and t == None:
                return False, 'time'

            elif m == None or d == None or y == None or t == None:
                return False, None
            time_minutes = 30 if '0.5' in str(time_of_day) else 0
            dt = datetime.datetime(int(year), m + 1, day, int(time_of_day), time_minutes)
            timestamp = time.mktime(dt.timetuple())
            if timestamp < time.time():
                print "This time is in the past"
                return False, None
            return True, None

        print month, day, year, time_of_day

        if is_valid_id(month, day, year, time_of_day)[0] == False:
            if is_valid_id(month, day, year, time_of_day)[1] == 'time':
                self.speech.play_audio([join(self.audio_vocab_dir, 'new_profile/enhanced/time_of_day.mp3'),
                                        join(self.audio_vocab_dir, 'new_profile/enhanced/sorry_i_couldnt_get_your.mp3')])
            else:
                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/didnt_understand.mp3'))
            self.speech.play_audio(join(self.audio_vocab_dir, 'appointment_management/when_would_appt.mp3'))
            self.book_new_appointment(False, client=client)

        elif self._check_slot_availability(month, day, year, time_of_day)[0] is True:
            # Time slot is free
            self._book_appointment_in_db(month, day, year, time_of_day, client=client)
            # Will return its own check. Just need to do confirmation
            self._booking_confirmation(month, day, year, time_of_day, client=client)
            return 'successfully booked'
        else:
            # Cannot book appointment
            cannot_book = join(self.speech.audio_vocab_dir, 'appointment_management/time_slot_already_booked.mp3')
            print "Time slot already booked"
            self.speech.microphone_mgr.commands.put({'Command':'Start', 'Wav_path':cannot_book})
            self.speech.play_audio(cannot_book)

            self.book_new_appointment(False, client=client)

    def new_user_profile(self, name):
        # Ask permission to remember the person's face

        profile = self.get_template_profile()

        name = name.replace('_', ' ').split(' ')

        if len(name) == 2:
            profile['First_Name'] = name[0]
            profile['Last_Name'] = name[1]

        elif len(name) == 3:
            profile['First_Name'] = name[0]
            profile['Middle_Name'] = name[1]
            profile['Last_Name'] = name[2]

        """
        FIRST, MIDDLE, LAST
        #----------------------------
        """

        def get_dob(play_init=True, play_except=False):
            if play_except:
                self.speech.play_audio(join(self.audio_vocab_dir, 'personal/didnt_understand.mp3'))

            if play_init:
                self.speech.play_audio(join(self.audio_vocab_dir, 'new_profile/enhanced/whens_birth_date.mp3'))

            resp = self.speech.listen()
            return resp

        def get_missed(dayset, monthset, yearset):
            missed = []
            missed.append(join(self.audio_vocab_dir, 'new_profile/enhanced/sorry_i_couldnt_get_your.mp3'))
            if len(dayset) == 0:
                missed.append(join(self.audio_vocab_dir, 'new_profile/enhanced/birth_day.mp3'))
            if len(monthset) == 0:
                missed.append(join(self.audio_vocab_dir, 'new_profile/enhanced/birth_month.mp3'))
            if len(yearset) == 0:
                missed.append(join(self.audio_vocab_dir, 'new_profile/enhanced/birth_year.mp3'))
            if len(missed) > 2:
                missed.insert(-1, self.audio_vocab_dir + 'personal/and.mp3')
            missed.append(join(self.audio_vocab_dir, 'personal/can_you_tell_me_what_these_are.mp3'))
            return missed

        dob_dayset = set()
        dob_monthset = set()
        dob_yearset = set()

        # For now just get the last one
        resp = get_dob()
        while True:
            if resp == None:
                while resp == None:
                    resp = get_dob(True, True)

            dob_month, dob_day, dob_year, dob_time = self.parse(resp, 'Date of Birth')
            [dob_dayset.add(i) for i in dob_day]
            [dob_monthset.add(i) for i in dob_month]
            [dob_yearset.add(i) for i in dob_year]

            if len(dob_dayset) > 0 and len(dob_monthset) > 0 and len(dob_yearset) > 0:
                break
            else:
                missed = get_missed(dob_dayset, dob_monthset, dob_yearset)
                self.speech.play_audio(missed)
                resp = get_dob(True, True)

        profile['DOB_Day'] = list(dob_dayset)[-1]
        profile['DOB_Month'] =  list(dob_monthset)[-1]
        profile['DOB_Month'] =  list(dob_yearset)[-1]

        self.speech.play_audio(join(self.audio_vocab_dir, 'personal/thanks.mp3'))

        """
        #----------------------------
        ADDRESS
        """

        def get_address(play_initializer=True):
            if play_initializer:
                # self.speech.microphone_mgr.commands.put({'Command' : 'Start', 'Wav_path' : join(self.audio_vocab_dir, 'new_profile/enhanced/whats_yer_address.mp3')})
                self.speech.play_audio(self.audio_vocab_dir + 'new_profile/enhanced/whats_yer_address.mp3')
            resp = self.speech.listen()
            return resp

        def get_missed(numberset, streetset, cityset, provinceset):
            missed = []
            missed.append(join(self.audio_vocab_dir, 'new_profile/enhanced/sorry_i_couldnt_get_your.mp3'))
            if len(numberset) == 0:
                missed.append(join(self.audio_vocab_dir, 'new_profile/enhanced/house_number.mp3'))
            if len(streetset) == 0:
                missed.append(join(self.audio_vocab_dir ,'new_profile/enhanced/street_name.mp3'))
            if len(cityset) == 0:
                missed.append(join(self.audio_vocab_dir, 'new_profile/enhanced/city.mp3'))
            if len(provinceset) == 0:
                missed.append(join(self.audio_vocab_dir, 'new_profile/enhanced/province.mp3'))
            if len(missed) > 2:
                missed.insert(-1, join(self.audio_vocab_dir, 'personal/and.mp3'))
            missed.append(join(self.audio_vocab_dir, 'personal/can_you_tell_me_what_these_are.mp3'))
            return missed

        numberset = set(); streetset = set(); cityset = set(); provinceset = set()
        resp = get_address()

        while True:
            if resp == None:
                while resp == None:
                    self.speech.play_audio(self.audio_vocab_dir + 'personal/didnt_understand.mp3')
                    resp = get_address(True)

            number, street, city, province = self.parse(resp, question='Home Address')
            print number, street, city, province
            [numberset.add(i) for i in number]; [streetset.add(i) for i in street]; [cityset.add(i) for i in city]; [provinceset.add(i) for i in province]

            if len(numberset) > 0 and len(streetset) > 0 and len(cityset) > 0 and len(provinceset) > 0:
                print len(numberset), len(streetset), len(cityset), len(provinceset)
                break
            else:
                missed = get_missed(numberset, streetset, cityset, provinceset)
                self.speech.play_audio(missed)
                resp = get_address(False)
                continue

        profile['Address_Number'] = list(numberset)[-1]
        profile['Address_Street'] = list(streetset)[-1]
        profile['Address_City'] = list(cityset)[-1]
        profile['Address_Province'] = list(provinceset)[-1]

        self.speech.play_audio(self.audio_vocab_dir + 'personal/thanks.mp3')

        """
        #----------------------------
        Phone Number
        """

        def get_phone(play_init=True, play_exception=False):
            if play_exception:
                self.speech.play_audio(join(self.audio_vocab_dir, 'new_profile/enhanced/my_mistake_areacode_and_7_digit_number.mp3'))
            if play_init:
                self.speech.play_audio(join(self.audio_vocab_dir, 'new_profile/enhanced/whats_yer_phone_number.mp3'))

            resp = self.speech.listen()
            return resp

        resp = get_phone()
        while True:
            if resp == None:
                while resp == None:
                    resp = get_phone(True, True)

            areacode, number = self.parse(resp, question='Phone Number') #get_phone()
            if areacode != None and number != None and len(number) == 7:
                break
            else:
                resp = get_phone(True, True)

                continue

        profile['Telephone_Number'] = areacode + number

        # Add the profile to the dictionary
        self.insert_new_profile(profile_record=profile)
        # Add an item to the queue so that

    def parse(self, user_str, question):
        # Assume no punctuation and split by spaces
        tokens = [i.lower() for i in user_str.split(' ')]
        outp = []

        if question == 'Name':
            for tok in tokens:
                if tok in self.translator.name_dict:
                    outp.append(tok)
            return outp

        elif question == 'Date of Birth':
            month, day, year, time = self.translator.get_date_time(user_str, return_clues=True, allow_defaults=False)
            print month, day, year, time
            return [month, day, year, time]

        elif question == 'Home Address':
            number, street, city, province = self.translator.get_address(user_str, return_clues=True)
            return [number, street, city, province]

        elif question == 'Phone Number':
            area_code, number = self.translator.get_phone_number(user_str)
            return [area_code, number]
        else:
            raise RuntimeError('Not a valid question')

    def run_new_user_form(self, application='enhanced', profile=None):
        """
        Assume depth of 2 and format as in self.new_user_profile
        """
        if profile is None:
            profile = self.new_user_profile(application)

        def open_and_ask(field, sub_field=False):
            prepend_dir = self.audio_vocab_dir + '/new_profile/'

            if 'ask' in profile[field].keys():
                if sub_field is False:
                    self.speech.play_audio(prepend_dir + profile[field]['ask'])
                else:
                    self.speech.play_audio(prepend_dir + profile[field][sub_field]['ask'])

        for field_category in profile.keys():
            fields = deepcopy(profile[field_category].keys())
            fields.remove('ask')

            open_and_ask(field_category)
            print(fields)
            for field in fields:
                yield open_and_ask(field_category, sub_field=field)

    """
    APPOINTMENT MANAGEMENT
    """

    def _booking_confirmation(self, month, day, year, time, client=None):
        """
        Confirm a booking date with the user
        :param month: (int)
        :param day: (int)
        :param time: (float)
        :return: (Audio) date confirmation
        """
        if client:
            wavs = [join(self.speech.audio_vocab_dir, 'personal/okay.mp3'),
                    join(self.speech.audio_vocab_dir, 'names/{}.mp3'.format(client.replace(' ', '_'))),
                join(self.speech.audio_vocab_dir, 'booking_confirm/booking_confirmation_for_name.mp3')
            ]
        else:
            wavs = [join(self.speech.audio_vocab_dir, 'personal/okay.mp3'),
                join(self.speech.audio_vocab_dir, 'booking_confirm/booking_confirmation_for_name.mp3')
            ]

        # Play an intro to the date confirmation
        for wv in wavs:
            self.speech.play_audio(wv)

        # There is an extension on time. Flatten out the lists
        say_list = []
        nested_say_lst = self.translator.get_date_speech(month, day, year, time)
        for sublst in nested_say_lst:
            if type(sublst) != list:
                say_list.append(sublst)
            else:
                say_list += sublst

        # Play each audio file to confirm date with the user
        for speech in say_list:
            self.speech.play_audio(speech)

        print('Appointment booked')

    def _check_slot_availability(self, month=None, day=None, year=None, time=None):
        """
        Determines whether the time slots provided are free
        :param Month: The month to be specified. Defaults as current month
        :param Day: The day to be specified. Defaults as current day
        :param Time: The time to be specified.
        :param path_to_db: The path to the appointment db
        :return: Whether the time slot(s) are free
        """
        if month is None:
            month = self.translator.current_month
        if day is None:
            day = self.translator.current_day
        if time is None:
            time = self.translator.current_time
        if year is None:
            time = self.translator.current_year

        print "time prior to snap is", time

        # Snap Time to closest time
        time = self.translator.snap_time(time)



        # Check the appointment database to see if the requested time is available
        valid_spots = self.appointment_db.search((self.db_query.Month == month) & (self.db_query.Day == day)
                                                 & (self.db_query.Time == time) & (self.db_query.Year == year) & (
                                                     self.db_query.Booked == False))

        print valid_spots

        if len(valid_spots) >= 1:
            # There are available times
            return True, (month, day, year, time)
        else:
            # There are not available times
            print "No valid entries!"
            return False, (month, day, year, time)


    def lookup_persons_next_appointment(self, client):
        try:
            current_epoch_time = int(time.time())

            valid_spots = self.appointment_db.search((self.db_query.Booked == True) &
                                                     (self.db_query.Client == client.replace('_', ' ')) &
                                                     (self.db_query.Timestamp >= current_epoch_time))

            sorted_appointments_by_epoch = sorted(valid_spots, key=lambda k: k['Timestamp'])

            return sorted_appointments_by_epoch

        except Exception as e:
            print e
            return None

    def _book_appointment_in_db(self, month, day, year, time, Doctor=None, Reason=None, Comment=None, client=None):
        """
        Assume time slot has already been checked and is available
        Book an apointment for 1 time slot
        :param month: Month to book appointment
        :param day: Day of month
        :param time: Time of day
        :return: Updates appintment db of booking
        """

        print "BOOKING APPT IN DB"
        # This time slot is free, proceed with booking
        self.appointment_db.update({'Booked': True}, (self.db_query.Month == month) &
                                   (self.db_query.Day == day) & (self.db_query.Year == year) & (self.db_query.Time == time))

        if client == None:
            raise RuntimeError("Client cannot be none!!")
        if client:
            self.appointment_db.update({"Client": client.replace('_', ' ')}, (self.db_query.Month == month) &
                                   (self.db_query.Day == day) & (self.db_query.Year == year) & (self.db_query.Time == time))


        # Additional fields that could be supplied/requested, etc.
        additional_fields = ['Doctor', 'Reason', 'Comment']
        additional_vals = [Doctor, Reason, Comment]
        # Update the DB with new information
        for field, val in zip(additional_fields, additional_vals):
            if not field:
                self.appointment_db.update(dbset(str(field), val), (self.db_query.Month == month) &
                                           (self.db_query.Day == day) & (self.db_query.Time == time))

    def get_template_profile(self):
        template_profile_record = {'First_Name': None,
                                   'Last_Name': None,
                                   'Middle_Name': None,

                                   'DOB_Day': None,
                                   'DOB_Month': None,
                                   'DOB_Year': None,

                                   'Address_Number': None,
                                   'Address_Street': None,
                                   'Address_City': None,
                                   'Address_Province': None,
                                   'Address_Postal Code': None,

                                   'Telephone_Number': None,
                                   'Email_Address': None,

                                   'Doctor': None
                                   }
        return template_profile_record

    def insert_new_profile(self, profile_record):
        from tinydb import TinyDB, Query

        path_to_db = join(self.ARBIE_dir, 'database/databases/client_profiles.json')
        db = TinyDB(path_to_db)
        db.insert(profile_record)

    def lookup_record(self, profile_record, specific_key='first'):
        from tinydb import TinyDB, Query
        pr = profile_record
        valid_spots = self.profile_db.search(
            (self.db_query.First_Name == pr['First_Name'])
            & (self.db_query.Middle_Name == pr['Middle_Name'])
            & (self.db_query.Last_Name == pr['Last_Name'])
            & (self.db_query.DOB_Day == pr['DOB_Day'])
            & (self.db_query.DOB_Month == pr['DOB_Month'])
            & (self.db_query.DOB_Year == pr['DOB_Year'])
            & (self.db_query.Address_Number == pr['Address_Number'])
            & (self.db_query.Address_Street == pr['Address_Street'])
            & (self.db_query.Address_Province == pr['Address_Province'])
            & (self.db_query.Address_Postal_Code == pr['Address_Postal_Code'])
            & (self.db_query.Telephone_Number == pr['Telephone_Number'])
            & (self.db_query.Email_Address == pr['Email_Address'])
            & (self.db_query.Doctor == pr['Doctor'])
        )
        return valid_spots

if __name__ == "__main__":
    rb = ARBIE(True, offline=500)
    rb.watch_and_listen(send_to_pi=True)
