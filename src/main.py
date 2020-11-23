import argparse
import pickle

from data_helper import DataHelper
from eval.evaluation import Evaluator
from models.classifiers import NeuralClassifier

from ubc_coref.trainer import Trainer
from ubc_coref.coref_model import CorefScore
from ubc_coref import loader

from models.parser_coref import NeuralRstParserCoref
from features.rst_dataset import RstDatasetCoref

import torch
from torch.utils.data import DataLoader
from utils.constants import *
from utils.other import collate_samples
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true',
                        help='whether to extract feature templates, action maps and relation maps')
    parser.add_argument('--train', action='store_true',
                        help='whether to train new models')
    parser.add_argument('--eval', action='store_true',
                        help='whether to do evaluation')
    parser.add_argument('--train_dir', help='train data directory')
    parser.add_argument('--eval_dir', help='eval data directory')
    parser.add_argument('--model_type', help='baseline/coref_feats/multitask/multitask-plain - 0/1/2/3')
    parser.add_argument('--model_name', help='Name of the model')
    parser.add_argument('--pretrained_coref_path', help='Path to the pretrained coref model')

    return parser.parse_args()

def get_train_loader(data_helper, config):
    action_feats = data_helper.feats_list
    action_labels = list(zip(data_helper.actions_numeric, 
                        data_helper.relations_numeric))    
    
        
    train_data = RstDatasetCoref(action_feats, action_labels, data_helper, is_train=True)
    
    train_loader = DataLoader(train_data, 
                               batch_size=config[BATCH_SIZE], 
                               shuffle=True,
                               collate_fn=lambda x: collate_samples(data_helper, x), 
                               drop_last=False)
    
    return train_loader


def get_coref_resolver(config):
    
    if config[MODEL_TYPE] > 0:
        coref_model = CorefScore(higher_order=True).to(config[DEVICE])
        if config[MODEL_TYPE] in [2, 3]:
            max_segment_len = 384
            train_corpus = pickle.load(open('../data/train_corpus_' + str(max_segment_len) + '.pkl', 'rb'))
            val_corpus = pickle.load(open('../data/val_corpus_' + str(max_segment_len) + '.pkl', 'rb'))
            test_corpus = pickle.load(open('../data/test_corpus_' + str(max_segment_len) + '.pkl', 'rb'))
            coref_trainer = Trainer(coref_model, train_corpus, val_corpus, 
                                    test_corpus, debug=False, config[PRETRAINED_COREF_PATH])
        else:
            coref_trainer = Trainer(coref_model, [], [], [], debug=False)
    else:
        coref_trainer = None
        
    return coref_trainer


def get_discourse_parser(data_helper, config):
    clf = NeuralClassifier(data_helper, config)
    clf.eval()
    clf.to(config[DEVICE])
    coref_trainer = get_coref_resolver(config)
    rst_parser = NeuralRstParserCoref(clf, coref_trainer, data_helper, config)
    return rst_parser


def train_model_coref(data_helper, config):
    
    helper_name = "data_helper_rst.bin"
    data_helper.load_data_helper(os.path.join('../data/', helper_name))
    train_loader = get_train_loader(data_helper, config)
    
    rst_parser = get_discourse_parser(data_helper, config)
    rst_parser.train_classifier(train_loader)
    
    
if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="3"
    args = parse_args()
    config = {
        OP_FEATS: False,
        ORG_FEATS: True,
        HIDDEN_DIM: 512,
        BATCH_SIZE: 5,
        DEVICE: "cuda:0",
        KEEP_BOUNDARIES: False,
        DO_COREF: True,
        MODEL_TYPE: int(args.model_type),
        MODEL_NAME: args.model_name,
        PRETRAINED_COREF_PATH: args.pretrained_coref_path
    }
    
    data_helper = DataHelper()    
    helper_name = "data_helper_rst.bin"
    helper_path = os.path.join('../data/', helper_name)
    
    if args.prepare:
        # Create training data
        coref_model = CorefScore(higher_order=True).to(config[DEVICE])
        
        coref_trainer = Trainer(coref_model, [], [], [], debug=False)
        
        data_helper.create_data_helper(args.train_dir, config, coref_trainer)
        data_helper.save_data_helper(helper_path)
            
    if args.train:
        train_model_coref(data_helper, config)
    
    if args.eval:
        # Evaluate models on the RST-DT test set
        data_helper.load_data_helper(helper_path)
        
        parser = get_discourse_parser(data_helper, config)
        parser.load('../data/model/' + config[MODEL_NAME])
        print("Evaluating")
        with torch.no_grad():
            evaluator = Evaluator(parser, data_helper, config)
            evaluator.eval_parser(None, path=args.eval_dir)
