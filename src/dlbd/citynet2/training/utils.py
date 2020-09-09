import os
import pickle
import traceback
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import yaml
from scipy.ndimage.interpolation import zoom

from lib.data_helpers import load_annotations
from lib.train_helpers import SpecSampler, create_net
import tf_slim as slim


def get_base_dir(opts, train=True):
    base = opts["base_dir"]
    if train:
        base += opts["train_dir"]
    else:
        base += opts["test_dir"]
    return base


def create_detection_dataset(opts, train=True):
    base = get_base_dir(train)
    dest_dir = base + opts["dest_dir"]
    audio_dir = base + opts["audio_dir"]
    labels_dir = base + opts["annotations_dir"]

    # load in the annotations
    save_dir = Path(dest_dir + opts.get("spec_type", "mel") + "/")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    for fname in os.listdir(audio_dir):
        savename = save_dir / fname.replace(".wav", ".pkl")

        # load the annottion
        try:
            if not os.path.exists(savename):
                annots, wav, sample_rate = load_annotations(
                    fname, audio_dir, labels_dir
                )

                spec = generate_spectrogram(wav, sample_rate, opts)

                # save to disk
                with open(savename, "wb") as f:
                    pickle.dump((annots, spec), f, -1)
            else:
                print("Skipping " + str(savename))
        except Exception:
            print("Error loading: " + fname + ", skipping.")
            print(traceback.format_exc())


def generate_spectrogram(wav, sample_rate, spec_opts):

    if spec_opts["spec_type"] == "mel":
        spec = librosa.feature.melspectrogram(
            wav,
            sr=sample_rate,
            n_fft=spec_opts.get("n_fft", 2048),
            hop_length=spec_opts.get("hop_length", 1024),
            n_mels=spec_opts["n_mels"],
        )
        spec = spec.astype(np.float32)
    else:
        raise AttributeError("No other spectrogram supported yet")
    return spec


# def generate_spectrograms(
#     src_dir, dest_dir, spec_opts=None,
# ):
#     spec_opts = spec_opts or {
#         "spec_type": "mel",
#         "n_fft": 2048,
#         "n_mels": 32,
#         "hop_length": 1024,
#     }
#     files = os.listdir(src_dir)

#     save_dir = dest_dir + spec_opts.get("spec_type", "mel") + "/"
#     for fname in files:
#         if fname.endswith(".pkl"):
#             dest_file = save_dir + fname
#             if not os.path.exists(dest_file):
#                 with open(src_dir + fname, "rb") as f:
#                     _, wav, sample_rate = pickle.load(f)

#                 print("Generating spectrogram file: ", dest_file)
#                 spec = librosa.feature.melspectrogram(
#                     wav,
#                     sr=sample_rate,
#                     n_fft=spec_opts.get("n_fft", 2048),
#                     hop_length=spec_opts.get("hop_length", 1024),
#                     n_mels=spec_opts["n_mels"],
#                 )
#                 spec = spec.astype(np.float32)

#                 with open(dest_file, "wb") as f:
#                     pickle.dump(spec, f, -1)
#             else:
#                 print("Skipping ", dest_file)
#     return save_dir


# def create_folds(spec_dir):
#     files = [xx for xx in os.listdir(spec_dir) if xx.endswith(".pkl")]
#     file_sites = [xx.split("_")[0] for xx in files]
#     print(len(files))
#     print(len(set(file_sites)))

#     site_counts = Counter(file_sites)
#     print(site_counts)

#     num_folds = 3

#     for seed in range(1000):
#         print("Seed is " + str(seed))
#         sites = sorted(list(set(file_sites)))
#         random.seed(seed)
#         random.shuffle(sites)

#         fold_size = int(len(sites) / num_folds)
#         file_fold_size = len(files) / num_folds

#         # manually getting the 3 folds
#         folds = []
#         folds.append(sites[:fold_size])
#         folds.append(sites[fold_size : 2 * fold_size])
#         folds.append(sites[2 * fold_size : 3 * fold_size])

#         wav_folds = []

#         passed = True

#         for fold in folds:
#             wav_fold_list = [
#                 xx.split("-sceneRect.csv")[0]
#                 for xx in files
#                 if xx.split("_")[0] in fold
#             ]
#             wav_folds.append(wav_fold_list)

#             num_files = sum([site_counts[xx] for xx in fold])
#             print(len(fold))
#             print(num_files)
#             if num_files < 6:
#                 passed = False
#         if passed:
#             break

#     # saving the folds to disk
#     savedir = base + "splits/"

#     print("Code commented out to prevent accidently overwriting")

#     # savepath = savedir + 'fold_sites.yaml'
#     # yaml.dump(folds, open(savepath, 'w'), default_flow_style=False)

#     # savepath = savedir + 'folds.yaml'
#     # yaml.dump(wav_folds, open(savepath, 'w'), default_flow_style=False)


def load_data_helper(file_name, opts):

    annots, spec = pickle.load(open(file_name, "rb"))
    annots = annots[opts["classname"]]
    # reshape annotations
    factor = float(spec.shape[1]) / annots.shape[0]
    annots = zoom(annots, factor)
    # create sampler
    if not opts["learn_log"]:
        spec = np.log(opts["A"] + opts["B"] * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)

    return annots, spec


def load_data(opts, train=True):
    # load data and make list of specsamplers
    X = []
    y = []

    base_dir = get_base_dir(opts, train)
    src_dir = base_dir + opts["dest_dir"]
    for file_name in os.listdir(src_dir):
        annots, spec = load_data_helper(file_name, opts)
        X.append(spec)
        y.append(annots[opts["classname"]])

    height = min(xx.shape[0] for xx in X)
    X = [xx[-height:, :] for xx in X]

    return X, y


def force_make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


def train_and_test(
    train_X,
    test_X,
    train_y,
    test_y,
    test_files,
    logging_dir,
    opts,
    TEST_FOLD=99,
    val_X=None,
    val_y=None,
):
    """
    Doesn't do any data loading - assumes the train and test data are passed
    in as parameters!
    """
    if val_X is None:
        val_X = test_X
        val_y = test_y

    # # creaging samplers and batch iterators
    train_sampler = SpecSampler(
        64,
        opts["HWW_X"],
        opts["HWW_Y"],
        opts["do_augmentation"],
        opts["learn_log"],
        randomise=True,
        balanced=True,
    )
    test_sampler = SpecSampler(
        64,
        opts["HWW_X"],
        opts["HWW_Y"],
        False,
        opts["learn_log"],
        randomise=False,
        seed=10,
        balanced=True,
    )

    height = train_X[0].shape[0]
    net = create_net(
        height,
        opts["HWW_X"],
        opts["HWW_Y"],
        opts["num_filters"],
        opts["wiggle_room"],
        opts["conv_filter_width"],
        opts["num_dense_units"],
        opts["do_batch_norm"],
    )

    y_in = tf.compat.v1.placeholder(tf.int32, (None))
    x_in = net["input"]

    print("todo - fix this up...")
    trn_output = net["fc8"]
    test_output = net["fc8"]

    _trn_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=trn_output, labels=y_in)
    )
    _test_loss = tf.compat.v1.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_output, labels=y_in)
    )
    print(y_in, trn_output, tf.argmax(trn_output, axis=1))

    pred = tf.cast(tf.argmax(trn_output, axis=1), tf.int32)
    _trn_acc = tf.reduce_mean(tf.cast(tf.equal(y_in, pred), tf.float32))

    pred = tf.cast(tf.argmax(test_output, axis=1), tf.int32)
    _test_acc = tf.reduce_mean(tf.cast(tf.equal(y_in, pred), tf.float32))

    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=opts.LEARNING_RATE, beta1=0.5, beta2=0.9
    )

    train_op = slim.learning.create_train_op(_trn_loss, optimizer)

    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(opts.MAX_EPOCHS):

            ######################
            # TRAINING
            trn_losses = []
            trn_accs = []

            for xx, yy in tqdm(train_sampler(train_X, train_y)):
                trn_ls, trn_acc, _ = sess.run(
                    [_trn_loss, _trn_acc, train_op], feed_dict={x_in: xx, y_in: yy}
                )
                trn_losses.append(trn_ls)
                trn_accs.append(trn_acc)

            ######################
            # VALIDATION
            val_losses = []
            val_accs = []

            for xx, yy in test_sampler(test_X, test_y):
                val_ls, val_acc = sess.run(
                    [_test_loss, _test_acc], feed_dict={x_in: xx, y_in: yy}
                )
                val_losses.append(val_ls)
                val_accs.append(val_acc)

            print(
                " %03d :: %02f  -  %02f  -  %02f  -  %02f"
                % (
                    epoch,
                    np.mean(trn_losses),
                    np.mean(trn_accs),
                    np.mean(val_losses),
                    np.mean(val_accs),
                )
            )

        #######################
        # TESTING
        results_savedir = force_make_dir(logging_dir + "results/")
        predictions_savedir = force_make_dir(logging_dir + "per_file_predictions/")

        test_sampler = SpecSampler(
            64,
            opts["HWW_X"],
            opts["HWW_Y"],
            False,
            opts["learn_log"],
            randomise=False,
            seed=10,
            balanced=False,
        )
        for fname, spec, y in zip(test_files, test_X, test_y):
            probas = []
            y_true = []
            for Xb, yb in test_sampler([spec], [y]):
                preds = sess.run(test_output, feed_dict={x_in: Xb})
                probas.append(preds)
                y_true.append(yb)

            y_pred_prob = np.vstack(probas)
            y_true = np.hstack(y_true)
            y_pred = np.argmax(y_pred_prob, axis=1)

            print("Saving to {}".format(predictions_savedir))
            with open(predictions_savedir + fname, "wb") as f:
                pickle.dump([y_true, y_pred_prob], f, -1)

        # save weights from network
        saver.save(sess, results_savedir + "weights_%d.pkl" % TEST_FOLD, global_step=1)


def train_citynet(opts):

    # X: spectrograms, y: labels
    train_X, train_y = load_data(opts, train=True)
    test_X, test_y = load_data(opts, train=False)

    test_files = os.listdir(get_base_dir(opts, train=False) + opts["dest_dir"])

    for idx in range(opts["ensemble_members"]):
        logging_dir = opts["base_dir"] + "predictions/%s/%d/%s/" % (
            opts["run_type"],
            idx,
            opts["classname"],
        )
        force_make_dir(logging_dir)
        # sys.stdout = ui.Logger(logging_dir + "log.txt")

        opts.height = train_X[0].shape[0]
        with open(logging_dir + "network_opts.yaml", "w") as f:
            yaml.dump(opts, f, default_flow_style=False)

        train_and_test(train_X, test_X, train_y, test_y, test_files, logging_dir, opts)
