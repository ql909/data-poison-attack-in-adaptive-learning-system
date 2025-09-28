import pandas as pd
import csv
import os
import os
import tensorflow as tf
import numpy as np
from utils import DKT
from load_data import DKTData
import argparse


rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

# train_path = os.path.join('./data/', 'skill_id_train.csv')
# test_path = os.path.join('./data/', 'skill_id_test.csv')

network_config = {}
network_config['batch_size'] = 32
network_config['hidden_layer_structure'] = [200]
network_config['learning_rate'] = 0.01
network_config['keep_prob'] = 0.333
network_config['rnn_cell'] = rnn_cells["LSTM"]


use_dktp = False

if use_dktp:
    network_config['lambda_o'] = 0.1
    network_config['lambda_w1'] = 0.3
    network_config['lambda_w2'] = 3.0
else:
    network_config['lambda_o'] = 0.0
    network_config['lambda_w1'] = 0.0
    network_config['lambda_w2'] = 0.0


num_runs = 1
num_epochs = 100
batch_size = 32
keep_prob = 0.333

def prepare_data(dataset_name):
    if dataset_name in ["HamptonAlg", "HamptonAlg_new"]:
        defaults = {'order_id': 'actionid', 'user_id': 'student', 'skill_id': 'knowledge', 'correct': 'assessment'}
    elif dataset_name in ["Assistment_challenge", "Assistment_challenge_new"]:
        defaults = {'order_id': 'action_num', 'user_id': 'studentId', 'skill_id': 'knowledge', 'correct': 'correct'}
    else:
        return

    train_path, test_path = dataset_name + "_train.csv", dataset_name + "_test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    rename_map = {v: k for k, v in defaults.items()}

    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    train_df = train_df[list(defaults.keys())]
    test_df = test_df[list(defaults.keys())]

    print(test_df)

    user_ids_train = train_df.user_id.unique()
    user_ids_test = test_df.user_id.unique()

    sorted_train_df = train_df.sort_values("order_id")
    sorted_test_df = test_df.sort_values("order_id")

    with open(dataset_name + "_for_dkt_train.csv", "w", newline="") as f:
        writer = csv.writer(f,
                            delimiter=',',
                            quotechar="'",
                            quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')
        for id in user_ids_train:
            df = sorted_train_df[sorted_train_df.user_id == id]
            df = df.dropna(subset=["skill_id"])
            problems = [int(pid) for pid in df.skill_id]
            corrects = df["correct"].fillna(-1).astype(int).tolist()
            num_problems = len(problems)
            #     print (num_problems)
            #     print (problems)
            #     print (corrects)
            #     print ("============")
            writer.writerow([num_problems])
            writer.writerow(problems)
            writer.writerow([int(i) for i in corrects])

    with open(dataset_name + "_for_dkt_test.csv", "w", newline="") as f:
        writer = csv.writer(f,
                            delimiter=',',
                            quotechar="'",
                            quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')
        for id in user_ids_test:
            df = sorted_test_df[sorted_test_df.user_id == id]
            problems = [int(pid) for pid in df.skill_id]
            corrects = [int(corr) for corr in df.correct]
            num_problems = len(problems)
            #     print (num_problems)
            #     print (problems)
            #     print (corrects)
            #     print ("============")
            writer.writerow([num_problems])
            writer.writerow(problems)
            writer.writerow([int(i) for i in corrects])


def run_DKT_new(dataset_name):
    tf.reset_default_graph()

    train_file_name = dataset_name
    train_path = train_file_name + "_for_dkt_train.csv"

    test_path = dataset_name + "_for_dkt_test.csv"

    print(train_path)
    print(test_path)

    if not os.path.exists(train_path):
        prepare_data(dataset_name)

    save_dir_prefix = './dkt_models/' + dataset_name + "/"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    data = DKTData(train_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_test = data.test
    num_problems = data.num_problems

    dkt = DKT(sess, data_train, data_test, num_problems, network_config,
              num_runs=num_runs, num_epochs=num_epochs,
              save_dir_prefix=save_dir_prefix,
              keep_prob=keep_prob, logging=True, save=True)

    dkt.model.build_graph()
    dkt.run_optimization()
    # close the session
    sess.close()


def convert_data_dkt(dataset_name, attack_type, ratio):
    if dataset_name in ["HamptonAlg", "HamptonAlg_new"]:
        defaults = {'order_id': 'actionid', 'user_id': 'student', 'skill_id': 'knowledge', 'correct': 'assessment'}
    elif dataset_name in ["Assistment", "Assistment_challenge_new"]:
        defaults = {'order_id': 'action_num', 'user_id': 'studentId', 'skill_id': 'knowledge', 'correct': 'correct'}
    else:
        return

    train_file_name = dataset_name + "_" + attack_type + "_" + str(ratio)
    train_path = train_file_name + ".csv"
    train_df = pd.read_csv(train_path)

    rename_map = {v: k for k, v in defaults.items()}

    train_df = train_df.rename(columns=rename_map)
    train_df = train_df[list(defaults.keys())]

    user_ids_train = train_df.user_id.unique()
    sorted_train_df = train_df.sort_values("order_id")

    with open(train_file_name + "_for_dkt.csv", "w", newline="") as f:
        writer = csv.writer(f,
                            delimiter=',',
                            quotechar="'",
                            quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')
        for id in user_ids_train:
            df = sorted_train_df[sorted_train_df.user_id == id]
            df = df.dropna(subset=["skill_id"])
            problems = [int(pid) for pid in df.skill_id]
            # corrects = [int(corr) for corr in df.correct]
            corrects = df["correct"].fillna(-1).astype(int).tolist()
            num_problems = len(problems)
            #     print (num_problems)
            #     print (problems)
            #     print (corrects)
            #     print ("============")
            writer.writerow([num_problems])
            writer.writerow(problems)
            writer.writerow([i for i in corrects])


def run_DKT_dpa(dataset_name, attack_type, ratio):
    tf.reset_default_graph()

    train_file_name = dataset_name + "_" + attack_type + "_" + str(ratio)
    train_path = train_file_name + "_for_dkt.csv"

    print(train_path)

    if not os.path.exists(train_path):
        convert_data_dkt(dataset_name, attack_type, ratio)

    test_path = dataset_name + "_for_dkt_test.csv"
    save_dir_prefix = './dkt_models/' + train_file_name + "/"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    data = DKTData(train_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_test = data.test
    num_problems = data.num_problems

    dkt = DKT(sess, data_train, data_test, num_problems, network_config,
              num_runs=num_runs, num_epochs=num_epochs,
              save_dir_prefix=save_dir_prefix,
              keep_prob=keep_prob, logging=True, save=True)

    dkt.model.build_graph()
    dkt.run_optimization()
    # close the session
    sess.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type", type=str, default=None,
                        help="Attack type: poisoned, sequential_pattern, hint_abuse, or None for baseline")
    parser.add_argument("--poison_mode", type=int, default=None,
                        help="Poison ratio (e.g., 5, 25, 50). Only used if attack_type is set.")
    args = parser.parse_args()

    # Optional: quiet TF logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if args.attack_type is None:
        print("Running baseline (no attack)...")
        run_DKT_new("Assistment_challenge_new")
    else:
        if args.poison_mode is None:
            raise ValueError("You must provide --poison_mode when --attack_type is set")
        print(f"Running attack={args.attack_type}, poison_mode={args.poison_mode} ...")
        run_DKT_dpa("Assistment_challenge_new", args.attack_type, args.poison_mode)