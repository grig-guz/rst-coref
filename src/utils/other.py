import torch


class ParseError(Exception):
    """ Exception for parsing
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ActionError(Exception):
    """ Exception for illegal parsing action
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
    
def reverse_dict(dct):
    """ Reverse the {key:val} in dct to
        {val:key}
    """
    newmap = {}
    for (key, val) in dct.items():
        newmap[val] = key
    return newmap
    
def collate_samples(data_helper, sample_list):
    all_docs, all_indexes, all_action_feats, all_neural_feats, all_actions, all_relations = [], [], [], [], [], []
    for feats, action in sample_list:
        all_docs.append(data_helper.docs[feats[0]])
        all_indexes.append(feats[0])
        all_action_feats.append(feats[1][0])
        all_neural_feats.append(feats[1][1])
        all_actions.append(action[0])
        all_relations.append(action[1])
    all_clusters = [data_helper.all_clusters[i] for i in all_indexes]
    all_actions, all_relations = torch.stack(all_actions), torch.stack(all_relations)
    return all_docs, all_clusters, \
           all_action_feats, all_neural_feats, \
           all_actions, all_relations, \
           all_relations > 0

def cleanup_load_dict(model_save):
    if 'operational_feats.weight' in model_save['model_state_dict']:
        del model_save['model_state_dict']['operational_feats.weight']
        del model_save['model_state_dict']['t12_dep_type_feats.weight']
        del model_save['model_state_dict']['st_q_dep_type_feats.weight']
        del model_save['model_state_dict']['subtree_form_feats.weight']
        del model_save['model_state_dict']['edu_length_feats.weight']
        del model_save['model_state_dict']['sent_length_feats.weight']
        del model_save['model_state_dict']['edu_comp_feats.weight']
        del model_save['model_state_dict']['sent_comp_feats.weight']
        del model_save['model_state_dict']['stack_cat_feats.weight']
        del model_save['model_state_dict']['queue_cat_feats.weight']


class2rel = {
    'Attribution': ['attribution', 'attribution-e', 'attribution-n', 'attribution-negative'],
    'Background': ['background', 'background-e', 'circumstance', 'circumstance-e'],
    'Cause': ['cause', 'cause-result', 'result', 'result-e', 'consequence', 'consequence-n-e', 'consequence-n',
              'consequence-s-e', 'consequence-s'],
    'Comparison': ['comparison', 'comparison-e', 'preference', 'preference-e', 'analogy', 'analogy-e', 'proportion'],
    'Condition': ['condition', 'condition-e', 'hypothetical', 'contingency', 'otherwise'],
    'Contrast': ['contrast', 'concession', 'concession-e', 'antithesis', 'antithesis-e'],
    'Elaboration': ['elaboration-additional', 'elaboration-additional-e', 'elaboration-general-specific-e',
                    'elaboration-general-specific', 'elaboration-part-whole', 'elaboration-part-whole-e',
                    'elaboration-process-step', 'elaboration-process-step-e', 'elaboration-object-attribute-e',
                    'elaboration-object-attribute', 'elaboration-set-member', 'elaboration-set-member-e', 'example',
                    'example-e', 'definition', 'definition-e'],
    'Enablement': ['purpose', 'purpose-e', 'enablement', 'enablement-e'],
    'Evaluation': ['evaluation', 'evaluation-n', 'evaluation-s-e', 'evaluation-s', 'interpretation-n',
                   'interpretation-s-e', 'interpretation-s', 'interpretation', 'conclusion', 'comment', 'comment-e'],
    'Explanation': ['evidence', 'evidence-e', 'explanation-argumentative', 'explanation-argumentative-e', 'reason',
                    'reason-e'],
    'Joint': ['list', 'disjunction'],
    'Manner-Means': ['manner', 'manner-e', 'means', 'means-e'],
    'Topic-Comment': ['problem-solution', 'problem-solution-n', 'problem-solution-s', 'question-answer',
                      'question-answer-n', 'question-answer-s', 'statement-response', 'statement-response-n',
                      'statement-response-s', 'topic-comment', 'comment-topic', 'rhetorical-question'],
    'Summary': ['summary', 'summary-n', 'summary-s', 'restatement', 'restatement-e'],
    'Temporal': ['temporal-before', 'temporal-before-e', 'temporal-after', 'temporal-after-e', 'temporal-same-time',
                 'temporal-same-time-e', 'sequence', 'inverted-sequence'],
    'Topic-Change': ['topic-shift', 'topic-drift'],
    'Textual-Organization': ['textualorganization'],
    'span': ['span'],
    'Same-Unit': ['same-unit']
}

val_trees = ['../data/data_dir/train_dir/wsj_1141.out.dis', '../data/data_dir/train_dir/wsj_1320.out.dis', '../data/data_dir/train_dir/wsj_2341.out.dis', '../data/data_dir/train_dir/wsj_1184.out.dis', '../data/data_dir/train_dir/wsj_0600.out.dis', '../data/data_dir/train_dir/wsj_0626.out.dis', '../data/data_dir/train_dir/wsj_2348.out.dis', '../data/data_dir/train_dir/wsj_1100.out.dis', '../data/data_dir/train_dir/wsj_2381.out.dis', '../data/data_dir/train_dir/wsj_0628.out.dis', '../data/data_dir/train_dir/wsj_1344.out.dis', '../data/data_dir/train_dir/wsj_1111.out.dis', '../data/data_dir/train_dir/wsj_0620.out.dis', '../data/data_dir/train_dir/wsj_1196.out.dis', '../data/data_dir/train_dir/wsj_1151.out.dis', '../data/data_dir/train_dir/wsj_2343.out.dis', '../data/data_dir/train_dir/wsj_0631.out.dis', '../data/data_dir/train_dir/wsj_1150.out.dis', '../data/data_dir/train_dir/wsj_2327.out.dis', '../data/data_dir/train_dir/wsj_1180.out.dis', '../data/data_dir/train_dir/wsj_1161.out.dis', '../data/data_dir/train_dir/wsj_1318.out.dis', '../data/data_dir/train_dir/wsj_1997.out.dis', '../data/data_dir/train_dir/wsj_1355.out.dis', '../data/data_dir/train_dir/wsj_2322.out.dis', '../data/data_dir/train_dir/wsj_1104.out.dis', '../data/data_dir/train_dir/wsj_1392.out.dis', '../data/data_dir/train_dir/wsj_1992.out.dis', '../data/data_dir/train_dir/wsj_1168.out.dis', '../data/data_dir/train_dir/wsj_1379.out.dis', '../data/data_dir/train_dir/wsj_1303.out.dis', '../data/data_dir/train_dir/wsj_2321.out.dis', '../data/data_dir/train_dir/wsj_2325.out.dis', '../data/data_dir/train_dir/wsj_0633.out.dis', '../data/data_dir/train_dir/wsj_1375.out.dis']

relation_map = {None: 0, 'Elaboration': 1, 'Same-Unit': 2, 'Attribution': 3, 'Contrast': 4, 'Cause': 5, 'Enablement': 6, 'Explanation': 7, 'Evaluation': 8, 'Temporal': 9, 'Manner-Means': 10, 'Joint': 11, 'Topic-Change': 12, 'Background': 13, 'Condition': 14, 'Comparison': 15, 'Summary': 16, 'Topic-Comment': 17, 'Textual-Organization': 18}

action_map = {('Shift', None): 0, ('Reduce', 'NS'): 1, ('Reduce', 'NN'): 2, ('Reduce', 'SN'): 3}

xidx_action_map, xidx_relation_map = reverse_dict(action_map), reverse_dict(relation_map)

rel2class = {}
for cl, rels in class2rel.items():
    rel2class[cl.lower()] = cl
    for rel in rels:
        rel2class[rel.lower()] = cl
