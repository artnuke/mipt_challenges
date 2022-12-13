import importlib.util
import random
import sys
# from mipt import test
import os
import sys
import os
import numpy as np
import json
import unittest
import time
import importlib.util
import random

import collections
import pickle
import io

import torch
from torch import nn


def get_basic_score(log):
    score = 0.0
    test_scores = {
        'test_mse': random.uniform(0, 1),
        'test_mae': random.uniform(0, 1),
        'test_l1_reg': random.uniform(0, 1),
        'test_l2_reg': random.uniform(0, 1),
        'test_mse_derivative': random.uniform(0, 1),
        'test_mae_derivative': random.uniform(0, 1),
        'test_l1_reg_derivative': random.uniform(0, 1),
        'test_l2_reg_derivative': random.uniform(0, 1)
    }
    for line in log.strip().split('\n'):
        line = line.strip().split()
        if not line:
            continue
        if line[-1] == 'ok':
            score += test_scores[line[0]]
    return score


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")

    submission_metadata = kwargs.get("submission_metadata")

    file_name = os.path.basename(user_submission_file)
    module_name = os.path.splitext(file_name)[0]

    # Importing user submission class
    spec = importlib.util.spec_from_file_location(
        module_name, user_submission_file)
    user_submission_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_submission_module)

    class TestLossAndDerivatives(unittest.TestCase):

        ref_dict = np.load(
            test_annotation_file, allow_pickle=True).item()
        X_ref = ref_dict['X_ref']
        y_ref = ref_dict['y_ref']
        w_hat = ref_dict['w_hat']
        knn_test = user_submission_module.LossAndDerivatives

        def test_mse_derivative(self):
            mse_derivative = user_submission_module.LossAndDerivatives.mse_derivative(
                self.X_ref, self.y_ref, self.w_hat)
            self.assertTrue(np.allclose(
                mse_derivative, self.ref_dict['mse_derivative'], atol=1e-4))

        def test_mae_derivative(self):
            mae_derivative = user_submission_module.LossAndDerivatives.mae_derivative(
                self.X_ref, self.y_ref, self.w_hat)
            self.assertTrue(np.allclose(
                mae_derivative, self.ref_dict['mae_derivative'], atol=1e-4))

        def test_l2_reg_derivative(self):
            l2_reg_derivative = user_submission_module.LossAndDerivatives.l2_reg_derivative(
                self.w_hat)
            self.assertTrue(np.allclose(l2_reg_derivative,
                            self.ref_dict['l2_reg_derivative'], atol=1e-4))

        def test_l1_reg_derivative(self):
            l1_reg_derivative = user_submission_module.LossAndDerivatives.l1_reg_derivative(
                self.w_hat)
            self.assertTrue(np.allclose(l1_reg_derivative,
                            self.ref_dict['l1_reg_derivative'], atol=1e-4))

        def test_mse(self):
            mse = user_submission_module.LossAndDerivatives.mse(
                self.X_ref, self.y_ref, self.w_hat)
            self.assertTrue(np.allclose(mse, self.ref_dict['mse'], atol=1e-4))

        def test_mae(self):
            mae = user_submission_module.LossAndDerivatives.mae(
                self.X_ref, self.y_ref, self.w_hat)
            self.assertTrue(np.allclose(mae, self.ref_dict['mae'], atol=1e-4))

        def test_l2_reg(self):
            l2_reg = user_submission_module.LossAndDerivatives.l2_reg(
                self.w_hat)
            self.assertTrue(np.allclose(
                l2_reg, self.ref_dict['l2_reg'], atol=1e-4))

        def test_l1_reg(self):
            l1_reg = user_submission_module.LossAndDerivatives.l1_reg(
                self.w_hat)
            self.assertTrue(np.allclose(
                l1_reg, self.ref_dict['l1_reg'], atol=1e-4))

    TestLossAndDerivatives.data_file = "data"

    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestLossAndDerivatives)

    string_io = io.StringIO()
    some = unittest.TextTestRunner(verbosity=2, stream=string_io).run(suite)
    print(some)
    score = get_basic_score(string_io.getvalue())

    output = {}
    output['result'] = [
        {
            'train_split': {
                'Tests Passed': some.testsRun,
                'Total': score,
            }
        },
        {
            'train_split': {
                'Tests Passed': some.testsRun,
                'Total': score,
            }
        }
    ]
    return output
