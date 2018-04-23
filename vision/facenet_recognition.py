import tensorflow as tf
import cv2
import os
from vision import detect_face
from vision import facenet
import cPickle as pickle
from sklearn.neural_network import MLPClassifier
import numpy as np
import calendar
import socket
import time
from os.path import dirname, abspath
from threading import Thread
import shutil
import datetime
from os.path import join


class FacialRecognition(object):
    def __init__(self):
        self.parent_dir = dirname(abspath(__file__)) + '/'
        self.audio_vocab_dir = dirname(dirname(self.parent_dir)) + '/database/databases/audio_vocabulary/'
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]   # three steps's self.threshold
        self.factor = 0.709  # scale self.factor

        self.model_dir = self.parent_dir + '20170512-110547'
        self.model_def = 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
        self.image_size = 96  # Image size (height, width) in pixels
        self.pool_type = 'MAX'  #{'MAX', 'L2'}.
        self.use_lrn = False  # Local Response Layer
        self.seed = 42
        self.batch_size = None

        # TODO: Turn this into a database to allow for restarts
        self.global_classifier_schedular = [23.9] # Retrain global classifier once at ~ midnight

        names_dir = self.audio_vocab_dir + 'names/'
        client_names = [i.split('.')[0].replace('_', ' ')for i in os.listdir(names_dir)]

        self.last_seen = {}.fromkeys(client_names, 0)
        self.welcome_dict = {}
        for name in client_names:
            self.welcome_dict[name] = names_dir + name.replace(' ', '_') + '.mp3'

        clf_dir = join(self.parent_dir, 'model_check_point')
        try: # Load the global classifiers, if available
            self.global_classifier = pickle.load(open(join(clf_dir + 'global_classifier/global_clf.pkl', 'w')))
        except Exception:
            self.global_classifier = None  # There is no global classifier

        # Load any local classifiers available
        local_clf_dir = join(clf_dir, 'local_classifiers')
        local_clfs = [join(local_clf_dir, i) for i in os.listdir(local_clf_dir)]
        self.local_classifiers = {}
        for pth in local_clfs:
            with open(pth, 'r') as f:
                clf = pickle.load(f)
                name_key = pth.split('/')[-1].split('.')[0]
                self.local_classifiers[name_key] = clf

        # Local classifier parameters
        self.is_talking_to_person = False
        self.collect_images = False
        self.num_faces_to_collect = 100
        self.num_collected_faces = 0
        self.num_norm_transformations = 4
        self.num_imgs_for_classification = 15
        self.local_train_x = []
        self.local_train_y = []
        self.training_label = -1
        self.local_clf_threshold = 0.8

        self.pnet = None
        self.rnet = None
        self.onet = None
        self.facenetGraph = tf.Graph()
        self.facenetSess  = tf.Session()
        self.mtcnnGraph   = tf.Graph()
        self.mtcnnSess    = tf.Session()
        self.restore_mtcnn()
        self.load_facenet()

        self.executing_appointment = False
        self.new_user_profile = False

        self.time_between_recognition = 5

        self.interaction_end_time = 0

    def process_in_proximity(self, speech_synthesizer, translator, profile_fn, affirmation_fn, send_to_pi=True,
                             rec_per_n_det=2, queue=None, polly=None):

        video_capture = cv2.VideoCapture(0)

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
                facenet.load_model(self.parent_dir + 'model_check_point/20170512-110547.pb')

                # Check if iteration is a detection or recognition iteration
                iteration = 0
                while video_capture.isOpened():  # check!
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    ret, frame = video_capture.read()
                    def get_emb_data(closest_face, images_placeholder, embeddings, phase_train_placeholder, random_crop_flip=False):
                        face_position = closest_face.astype(int)
                        # Crop and align
                        crop = img[face_position[1]:face_position[3], face_position[0]:face_position[2], ]
                        # Scale
                        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                        # Reshape
                        data = crop.reshape(-1, 160, 160, 3)
                        # Preprocess data same as training
                        data = facenet.prewhiten(data)

                        if random_crop_flip:
                            data = facenet.crop(data, True, 160)
                            data = facenet.flip(data, True)

                        # Embed data
                        feed_dict = {images_placeholder: data, phase_train_placeholder: False}
                        emb_data = sess.run(embeddings, feed_dict=feed_dict)[0]
                        emb_data = np.array(emb_data.tolist()).reshape(1, -1)
                        return emb_data

                    try:
                        if ret:
                            speech_synthesizer.microphone_mgr.commands.put({'Command': 'Start', 'Wav_path': None})

                            # Get next frame
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            img = self.to_rgb(gray) if gray.ndim == 2 else gray
                            # Detect face(s) in frame
                            bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                                        self.threshold, self.factor)

                            # Only process the closest face
                            ranked_faces = self.rank_face_areas(bounding_boxes)
                            closest_face = ranked_faces[0][0]

                            if self.face_close(closest_face):
                                face_position = closest_face.astype(int)
                                cv2.rectangle(frame, (face_position[0], face_position[1]),
                                              (face_position[2], face_position[3]), (0, 255, 0), 2)

                                # Tracking iteration
                                if self.collect_images or (iteration % rec_per_n_det != 0):
                                    """
                                    Tracking Iteration
                                    """
                                    # Get information for face tracking.
                                    frame_height, frame_width = video_capture.read()[1].shape[:2]
                                    mdpt_coords = int(np.mean((face_position[2], face_position[0]))), int(np.mean(
                                        (face_position[3], face_position[1]))) # x, y Midpoint of Face
                                    ref_coords = int(frame_width / 2.), int(frame_height / 2.) # x, y Midpoint of Frame
                                    # Draw a triangle from frame midpoint to face midpoint and label
                                    x_line = (ref_coords[0], ref_coords[1]), (mdpt_coords[0], ref_coords[1])
                                    y_line = (mdpt_coords[0], ref_coords[1]), (mdpt_coords[0], mdpt_coords[1])
                                    cv2.line(frame, x_line[0], x_line[1], (0, 0, 255), 2)  # Delta x
                                    cv2.line(frame, y_line[0], y_line[1], (0, 0, 255), 2)  # Delta y
                                    x_coord, y_coord = mdpt_coords

                                    # Send information to the rasberry pi for tracking
                                    if send_to_pi:
                                        # Connect the socket to the port where the server is listening
                                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                        server_address = ('10.42.0.42', 10000)
                                        sock.connect(server_address)
                                        tuple_str = '{0} {1}'.format(x_coord, y_coord)
                                        sock.send(tuple_str)

                                    # If the person is a new face and we should remember them
                                    if self.collect_images:
                                        # Apply preprocessing
                                        for _ in range(self.num_norm_transformations):
                                            emb_data = get_emb_data(closest_face, images_placeholder, embeddings, phase_train_placeholder, random_crop_flip=True)
                                            self.local_train_x.append(emb_data[0].tolist())
                                            self.local_train_y.append(self.training_label)
                                        self.num_collected_faces += 1
                                        print self.num_collected_faces

                                        # End face collection process
                                        if self.num_collected_faces >= self.num_faces_to_collect:
                                            # Reset local classifier parameters
                                            print ("Retraining classifier")

                                            t = Thread(target=self.retrain_local_clf, args=(self.local_train_x, self.local_train_y))
                                            t.start()
                                            t.join()

                                            self.local_train_x = []
                                            self.local_train_y = []
                                            self.is_talking_to_person = False

                                            self.collect_images = False
                                            self.num_collected_faces = 0

                                            print 'added flag to queue'
                                            queue.put({'still_collecting_images': False})
                                            queue.put({'still_collecting_images': False})
                                            queue.put({'still_collecting_images': False})
                                            queue.put({'still_collecting_images': False})
                                            queue.put({'still_collecting_images': False})

                                """
                                Recognition Iteration
                                """

                                if queue.qsize() == 0 and not self.is_talking_to_person and \
                                        (time.time() - self.interaction_end_time > self.time_between_recognition) and iteration % rec_per_n_det == 0:

                                    # Don't let microphone adjust
                                    speech_synthesizer.microphone_mgr.played_audio.put(True)

                                    while speech_synthesizer.microphone_mgr.commands.qsize() > 0:
                                        speech_synthesizer.microphone_mgr.commands.get()

                                    # Recognition iteration
                                    local_clf_prediction_prob = {}
                                    global_clf_prediction_prob = {}

                                    id = 'unknown'  # By default, the person is unknown until proven otherwise
                                    # Collect multiple images & predict on each. Significant performance improvement
                                    for _ in range(self.num_imgs_for_classification):
                                        ret, frame = video_capture.read()
                                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                        img = self.to_rgb(gray) if gray.ndim == 2 else gray
                                        # Detect face(s) in frame
                                        bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet,
                                                                                    self.rnet, self.onet,
                                                                                    self.threshold, self.factor)
                                        # Only process the closest face
                                        ranked_faces = self.rank_face_areas(bounding_boxes)
                                        closest_face = ranked_faces[0][0]

                                        if self.face_close(closest_face):
                                            emb_data = get_emb_data(closest_face, images_placeholder, embeddings, phase_train_placeholder)
                                            face_position = closest_face.astype(int)

                                            # Local classifier
                                            for name, clf in self.local_classifiers.items():
                                                if name not in local_clf_prediction_prob:
                                                    local_clf_prediction_prob[name] = 0
                                                class_orders = list(clf.classes_)
                                                cls_idx = class_orders.index(self.training_label)
                                                pred_proba = clf.predict_proba(emb_data)[0]
                                                local_clf_prediction_prob[name.split('.')[0]] += pred_proba[cls_idx]

                                            # Global classifier
                                            if self.global_classifier:
                                                best_person = self.global_classifier.predict(emb_data)[0]
                                                if best_person not in global_clf_prediction_prob:
                                                    global_clf_prediction_prob[best_person] = 0
                                                proba = self.global_classifier.predict_proba(emb_data)[0]
                                                global_clf_prediction_prob[best_person] += proba

                                    # See if person in local clf
                                    # print 'local prob is:', local_clf_prediction_prob
                                    for name, unnormed_prob in local_clf_prediction_prob.items():
                                        if unnormed_prob / self.num_imgs_for_classification > self.local_clf_threshold:
                                            id = name.replace('_', ' ')
                                            break
                                    # See if person in global clf
                                    # print 'global prob is:', local_clf_prediction_prob
                                    # See if person in global clf
                                    for name, unnormed_prob in global_clf_prediction_prob.items():
                                        if unnormed_prob / self.num_imgs_for_classification > self.local_clf_threshold:
                                            id = name.replace('_', ' ')
                                            break

                                    print "Predicted ID is:", id
                                    # Print the name to a line on the screen
                                    text_x = face_position[0]
                                    text_y = face_position[3] + 20
                                    cv2.putText(frame, "ID: {0}".format(id), (text_x, text_y),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)

                                    # Only say hi to people every wait_time seconds
                                    wait_time = 1

                                    # The person identified is unknown.
                                    if id == 'unknown':
                                        speech_synthesizer.play_audio(self.welcome_dict[id])  # New user profile

                                        new_user_profile_affirmation = affirmation_fn()
                                        if new_user_profile_affirmation == None:
                                            while new_user_profile_affirmation == None:
                                                speech_synthesizer.play_audio(
                                                    self.audio_vocab_dir + 'personal/didnt_understand.mp3')
                                                speech_synthesizer.play_audio(self.welcome_dict[id])  # New user profile
                                                new_user_profile_affirmation = affirmation_fn()

                                        def get_person_name():

                                            def ask_person_for_name(speech_synthesizer, initial_hello=True,
                                                         play_exception=False):
                                                # Get name
                                                if initial_hello:
                                                    speech_synthesizer.play_audio(
                                                        self.audio_vocab_dir + 'new_profile/enhanced/whats_full_name.mp3')
                                                if play_exception:
                                                    speech_synthesizer.play_audio(
                                                        self.audio_vocab_dir + 'personal/didnt_understand.mp3')
                                                    speech_synthesizer.play_audio(
                                                        self.audio_vocab_dir + 'new_profile/enhanced/whats_full_name.mp3')
                                                resp = speech_synthesizer.listen()
                                                return resp

                                            resp = ask_person_for_name(speech_synthesizer, True, False)
                                            while True:
                                                if resp == None:
                                                    while resp == None:
                                                        resp = ask_person_for_name(speech_synthesizer, True, True)
                                                split_resp = [i for i in resp.split(' ') if len(i) > 1]
                                                if len(split_resp) < 2:
                                                    resp = ask_person_for_name(speech_synthesizer, True, True)
                                                    continue
                                                else:
                                                    break

                                            namelist = split_resp

                                            spoken_text = polly.synthesize_speech(Text=namelist[0],
                                                                                  OutputFormat='mp3',
                                                                                  VoiceId='Joanna')

                                            save_name_pth = self.audio_vocab_dir + 'names/' + str(
                                                namelist[0] + '_' + namelist[1]) + '.mp3'

                                            with open(save_name_pth, 'wb') as f:
                                                f.write(spoken_text['AudioStream'].read())

                                            # Reset audio duratoins dict
                                            os.system('rm {}'.format(join(dirname(dirname(self.parent_dir)),
                                                                                  'database/databases/audio_duration_dict.pkl')))

                                            speech_synthesizer.initialize_audio_dict()
                                            speech_synthesizer.audio_durations_dict = pickle.load(open(
                                                join(dirname(dirname(self.parent_dir)),
                                                             'database/databases/audio_duration_dict.pkl'), 'r'))

                                            new_greetings = [
                                                self.audio_vocab_dir + 'personal/hi.mp3',
                                                save_name_pth,
                                                self.audio_vocab_dir + 'personal/very_nice_to_meet_you.mp3'
                                            ]

                                            self.restore_namedict()

                                            for wav in new_greetings:
                                                speech_synthesizer.play_audio(wav)

                                            return namelist

                                        # If the person says yes to new user profile...
                                        if new_user_profile_affirmation:
                                            self.is_talking_to_person = True
                                            self.collect_images = True

                                            # Get the name of the person
                                            namelist = [i for i in get_person_name() if len(i) >= 2]

                                            # Launch a thread for the new user profile given the person's name as a starting point
                                            try:
                                                self.person_name = str(namelist[0]) + '_' + str(namelist[-1])
                                                assert len(self.person_name) >= 2
                                            except Exception:
                                                raise RuntimeError("Need to fix this! Didnt get name")

                                            speech_synthesizer.play_audio(self.audio_vocab_dir + 'personal/ask_a_few_questions.mp3')
                                            # Check for affirmation
                                            ask_a_few_questions = affirmation_fn()
                                            if ask_a_few_questions == None:
                                                while ask_a_few_questions == None:
                                                    speech_synthesizer.play_audio(
                                                        self.audio_vocab_dir + 'personal/didnt_understand.mp3')
                                                    speech_synthesizer.play_audio(
                                                        self.audio_vocab_dir + 'personal/ask_a_few_questions.mp3')
                                                    ask_a_few_questions =  affirmation_fn()

                                            if ask_a_few_questions:
                                                self.new_user_profile = True
                                                queue.put({'new_user_profile': 'execute', 'client': self.person_name, 'questions':True})
                                            else:
                                                Thread(target=speech_synthesizer.play_audio, args=(self.audio_vocab_dir + 'personal/no_worries_look_around_30_s.mp3',)).start()
                                                queue.put({'new_user_profile': 'execute', 'client': self.person_name, 'questions':False})
                                                self.new_user_profile = False
                                        else:
                                            speech_synthesizer.play_audio(join(self.audio_vocab_dir, 'personal/functionality_only_for_users.mp3'))

                                    else:
                                        # Check when the person was last seen
                                        print "entering into ID", id
                                        print self.welcome_dict

                                        try:
                                            speech_synthesizer.audio_durations_dict[self.welcome_dict[id]]
                                        except Exception as e:

                                            print e
                                            pass

                                        last_time_saw = self.last_seen[id]
                                        curr_time = calendar.timegm(time.gmtime())
                                        time_diff = (curr_time - last_time_saw)  # seconds

                                        if time_diff > wait_time:

                                            print "saying hi to name:", id

                                            new_greetings = [
                                                self.audio_vocab_dir + 'personal/hi.mp3',
                                                self.welcome_dict[id],
                                            ]

                                            speech_synthesizer.play_audio(new_greetings)

                                            # Check if the person has an appointment today
                                            queue.put({'check_for_appointment': 'execute', 'client': id})

                                            # Listen for possible commands
                                            self.last_seen[id] = curr_time

                                            self.is_talking_to_person = False

                                        else:
                                            # Don't say anything
                                            print ('Ignoring {0} since I saw them {1} seconds ago '.format(id, time_diff))

                                    speech_synthesizer.microphone_mgr.played_audio.get()
                                # Increment iterations

                                iteration += 1
                                # Show video
                                cv2.imshow('Video', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                else:
                                    pass

                    except Exception as e:
                        print e
                        continue

                video_capture.release()
                cv2.destroyAllWindows()

    def rank_face_areas(self, face_list):
        def get_face_area(face_position):
            dl = face_position[2] - face_position[0]
            dh = face_position[3] - face_position[1]
            return dl * dh

        if type(face_list) == None:
            return None

        else:
            area_list = map(get_face_area, face_list)
            ret_list = zip(face_list, area_list)
            ret_list.sort(key=lambda x: x[1], reverse=True)
            return ret_list

    def face_close(self, face_position, min_area=3000):
        def get_face_area(face_position):
            dl = face_position[2] - face_position[0]
            dh = face_position[3] - face_position[1]
            return dl * dh

        if type(face_position) != None:
            if get_face_area(face_position) > min_area:
                return True
            else:
                return False
        else:
            return False

    def retrain_local_clf(self, train_x, train_y):
        print('Beginning model training')

        # Dump the training data
        with open(join(self.parent_dir, 'train_dir/{0}.pkl'.format(self.person_name)), 'w') as f:
            pickle.dump([train_x, train_y], f)

        model = MLPClassifier((100, 50, 25))

        noise_x, noise_y = pickle.load(open(
            join(self.parent_dir, 'model_check_point/processed_noise_faces.pkl'), 'r'))

        all_x = noise_x + train_x
        all_y = noise_y + train_y

        model.fit(all_x, all_y)
        self.local_classifiers[self.person_name] = model

        print('Dumping learned results')

        print self.person_name

        with open(join(self.parent_dir, 'model_check_point/local_classifiers/{}.pkl'.format(self.person_name)),
                  'w') as f:
            pickle.dump(model, f)

        print('resetting parameters and adding welcome dict')
        names_dir = self.audio_vocab_dir + 'names/'
        client_names = [i.split('.')[0].replace('_', ' ') for i in os.listdir(names_dir)]

        self.welcome_dict = {}
        for name in client_names:
            try:
                if len(name.split('_')[1]) < 2:
                    name = name.split('_')[0]
            except Exception:
                pass
            self.welcome_dict[name] = names_dir + name.replace(' ', '_') + '.mp3'
        print 'Done training model'
        return

    def unique(self, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    def get_face_area(self, face_position):
        dl = face_position[2] - face_position[0]
        dh = face_position[3] - face_position[1]
        return dl * dh

    def to_rgb(self, img):
      w, h = img.shape
      ret = np.empty((w, h, 3), dtype=np.uint8)
      ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
      return ret

    def register_new_user(self, camera, person, num_samples_to_get=100):
        self.restore_mtcnn()
        person = person.replace(' ', '_')
        # restore facenet model
        print 'restoring facenet model'
        with self.facenetGraph.as_default():
            with self.facenetSess.as_default():
                self.load_facenet()
                video_capture = camera
                c = sample_num = 0

                # Check if the user exists. If not, add them
                if not os.path.exists(self.parent_dir + 'train_dir/{0}'.format(person)):
                    os.mkdir(self.parent_dir + 'train_dir/{0}'.format(person))

                print 'started scanning for faces'
                while (video_capture.isOpened()):  # check!
                    # Capture frame-by-frame
                    ret, frame = video_capture.read()
                    if ret != False:
                        # Get next frame
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        img = self.to_rgb(gray) if gray.ndim == 2 else gray
                        # Find faces
                        bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
                        # For each face..
                        for face_position in bounding_boxes:
                            try:
                                # Get face
                                face_position = face_position.astype(int)
                                cv2.rectangle(frame, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0, 255, 0), 2)
                                # Crop and resize
                                crop = img[face_position[1]:face_position[3], face_position[0]:face_position[2], ]
                                crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                                data = crop.reshape(-1, 160, 160, 3)
                                # Save
                                pickle.dump(data, open(self.parent_dir + "train_dir/{0}/{1}.pkl".format(person, sample_num), 'w'))
                                print('writing face image {0}'.format(sample_num))
                                # Draw a rectangle around the faces
                                cv2.putText(frame, 'Training on your face:', (50, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), thickness=2, lineType=2)
                            except Exception as e:
                                pass
                        c += 1
                        sample_num += 1
                        cv2.imshow('Video', frame)
                    if sample_num > num_samples_to_get:
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    pass

    def get_data(self, train_dir):
        predictor_mapper = {}
        data = {}

        # Only get non-processed data
        ignore_key = 'already_preprocessed'
        classes = [i for i in os.listdir(train_dir) if ignore_key not in i]

        encoding_class_idx = 0
        print('Number of people for reetraining is: {0}'.format(len(classes)))
        for cls in classes:
            data[cls] = []

            # List the contents of each class
            subdir = join(train_dir, cls)
            class_image_paths = [join(subdir, i) for i in os.listdir(subdir)]
            for img_path in class_image_paths:
                # Open pickle filea
                img = pickle.load(open(img_path, 'r'))
                data[cls].append(img)
            encoding_class_idx += 1

        # Move the trained data paths into the other folder
        already_processed_local_data = train_dir + '/already_preprocessed_data/local_data'
        print(already_processed_local_data)
        for cls in classes:
            pth = join(train_dir, cls)
            print('Path is: {0}, train dir is: {1}'.format(pth, already_processed_local_data))
            shutil.move(pth, already_processed_local_data)

        return data, predictor_mapper

    def local_retrain_classifier(self, individual_name):
        # This is a global retrainer for the classifier
        """
        The items (directories) in train_dir will be trained on.
        Previously processed training data is located in train_dir/already_preprocessed_data
        :return: None -- Retrained classifier in appropriate path
        """
        with tf.Graph().as_default():
            with tf.Session() as sess:
                timestamp = time.time()
                def recreate_training_checkpoint(timestamp):
                    model_path = self.parent_dir + 'model_check_point/20170512-110547.pb'
                    print('Loading feature extraction model')

                    facenet.load_model(model_path)

                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder =tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    print('Getting training data')
                    data, predictor_mapper = self.get_data(self.parent_dir + 'train_dir')
                    print("Finished getting training data. Training on new data")

                    train_x = []
                    train_y = []
                    total = 0
                    for cls in data.keys():
                        total += len(data[cls])

                    print('total iterations is: {0}'.format(total))
                    it = 0
                    for cls in data.keys():
                        images = data[cls]
                        for img in images:
                            # Apply preprocessing
                            img = facenet.prewhiten(img)
                            for _ in range(50):
                                img = facenet.crop(img, True, 160)
                                img = facenet.flip(img, True)
                                feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                                emb_data = sess.run(embeddings, feed_dict=feed_dict)[0]
                                train_x.append(emb_data.tolist())
                                train_y.append(cls)
                            it += 1
                            print('{} percent complete training'.format(float(it)/total * 100))

                    local_train_data_pkl = self.parent_dir + 'train_dir/already_preprocessed_data/train_data_{0}.pkl'.format(timestamp)
                    with open(local_train_data_pkl, 'wb') as f:
                        pickle.dump((train_x, train_y), f)
                    return local_train_data_pkl

                def rebuild_classifier(train_x, train_y, timestamp):
                    model = MLPClassifier((100, 50, 25))
                    model.fit(np.array(train_x), np.array(train_y))
                    local_classifier_filename_exp = self.parent_dir + 'model_check_point' + '/local_classifiers' \
                                                    + '/local_clf_{0}.pkl'.format(individual_name)

                    with open(local_classifier_filename_exp, 'wb') as f:
                        pickle.dump(model, f)

                local_train_data_pkl = recreate_training_checkpoint(timestamp)
                with open(local_train_data_pkl, 'rb') as f:
                    x_train, y_train = pickle.load(f)
                rebuild_classifier(x_train, y_train, timestamp)

    def global_retrain_classifier(self):
        # Local retrainer
        """
        The items (directories) in train_dir will be trained on.
        Previously processed training data is located in train_dir/already_preprocessed_data
        :return: None -- Retrained classifier in appropriate path
        """
        print 'Beginning global retraining'
        train_dir = join(self.parent_dir, 'train_dir')
        people_classifiers = os.listdir(train_dir)

        all_train_x = []
        all_train_y = []

        noise_data_x, noise_data_y = pickle.load(open(join(self.parent_dir, 'model_check_point/processed_noise_faces.pkl')))

        all_train_x += noise_data_x
        all_train_y += noise_data_y

        for ppl_clf in people_classifiers:
            train_x, _train_y = pickle.load(open(join(train_dir, ppl_clf)))
            train_y = [ppl_clf.split('.')[0]] * len(_train_y)
            all_train_x += train_x
            all_train_y += train_y

        model = MLPClassifier((100, 50, 25))
        model.fit(all_train_x,all_train_y)

        with open(join(self.parent_dir, 'model_check_point/global_classifier/global_clf.pkl'), 'w') as f:
            pickle.dump(model, f)

    def load_facenet(self):
        with self.facenetGraph.as_default():
            with self.facenetSess.as_default():
                model_path = self.parent_dir + 'model_check_point/20170512-110547.pb'
                print('Loading feature extraction model')
                facenet.load_model(model_path)

    def restore_mtcnn(self):
        # restore mtcnn model
        model_check_point = self.parent_dir + 'model_check_point'
        print('Creating networks and loading parameters')
        gpu_memory_fraction = 1.0
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, model_check_point)

    def process_noise_data(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                model_path = self.parent_dir + 'model_check_point/20170512-110547.pb'
                facenet.load_model(model_path)

                # Path to LFW dataset
                lfw_path = '/Users/Jonathan/git/ARBIE_Desk_Chatbot/vision/model_check_point/lfw'
                # Get raw jpgs
                noise_people = []
                for file in os.listdir(lfw_path):
                    imgs_in_file = os.listdir(join(lfw_path, file))
                    if len(imgs_in_file) < self.num_faces_to_collect:
                        continue
                    else:
                        img_class = []
                        for img_pth in imgs_in_file[0:self.num_faces_to_collect]:
                            img_class.append(join(file, img_pth))
                        noise_people.append(img_class)

                train_x = []
                train_y = []

                # Increment the label from 0 to n
                label = 0

                for person in noise_people:
                    for frame_pth in person:
                        try:
                            frame = cv2.imread(join(lfw_path,frame_pth), 1)

                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            img = self.to_rgb(gray) if gray.ndim == 2 else gray
                            # Detect face(s) in frame
                            bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                                        self.threshold, self.factor)

                            # Only process the closest face
                            ranked_faces = self.rank_face_areas(bounding_boxes)
                            closest_face = ranked_faces[0][0]

                            if self.face_close(closest_face):
                                face_position = closest_face.astype(int)

                                # Crop and resize
                                crop = img[face_position[1]:face_position[3],
                                       face_position[0]:face_position[2], ]
                                crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                                data = crop.reshape(-1, 160, 160, 3)

                                img = facenet.prewhiten(data)

                                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
                                    "phase_train:0")

                                # Apply preprocessing
                                for _ in range(self.num_norm_transformations):
                                    img = facenet.crop(img, True, 160)
                                    img = facenet.flip(img, True)
                                    feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                                    emb_data = sess.run(embeddings, feed_dict=feed_dict)[0]
                                    train_x.append(emb_data.tolist())
                                    train_y.append(label)
                        except Exception as e:
                            print(e)
                            pass

                    label += 1

                print'Dumping files'
                with open(join(self.parent_dir, 'model_check_point/processed_noise_faces.pkl'), 'w') as f:
                    pickle.dump([train_x, train_y], f)

    def scheduled_global_classifier(self):
        # Simple method that is ran at least every 5 minutes
        date = datetime.datetime.now()
        hr_since_midnight = date.hour + float(date.minute)/60
        if hr_since_midnight > self.global_classifier_schedular[0]:
            self.global_classifier_schedular.append(self.global_classifier_schedular.pop(0))
            print 'Next global update is:', self.global_classifier_schedular[0]

            self.global_retrain_classifier()

    def restore_namedict(self):
        self.welcome_dict = {}
        names_dir = self.audio_vocab_dir + 'names/'
        client_names = [i.split('.')[0].replace('_', ' ') for i in os.listdir(names_dir)]

        for name in client_names:
            self.welcome_dict[name] = names_dir + name.replace(' ', '_') + '.mp3'
            self.welcome_dict[name] = names_dir + name.split(' ')[0] + '.mp3'










