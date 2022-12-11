import random
import sys
import os


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    submission_metadata = kwargs.get("submission_metadata")
    sys.path.insert(1, 'user_submission_file')
    os.environ["test"] = test_annotation_file
    print(os.environ["test"])
    # score = test()
    score = 0
    output = {}
    output['result'] = [
        {
            'train_split': {
                'Metric1': 123,
                'Metric2': 123,
                'Metric3': 123,
                'Total': score,
            }
        },
        {
            'test_split': {
                'Metric1': 123,
                'Metric2': 123,
                'Metric3': 123,
                'Total': score,
            }
        }
    ]
    return output
