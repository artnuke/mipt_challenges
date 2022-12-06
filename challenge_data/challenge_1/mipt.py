#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np
import json
import unittest
import time

import collections
import pickle
import io

import torch
from torch import nn

from tested_assignment import create_model, count_parameters


def get_basic_score(log):
    score = 0.0
    test_scores = {
        'test_mse': 0.35,
        'test_mae': 0.35,
        'test_l1_reg': 0.15,
        'test_l2_reg': 0.15,
        'test_mse_derivative': 0.35,
        'test_mae_derivative': 0.35,
        'test_l1_reg_derivative': 0.15,
        'test_l2_reg_derivative': 0.15,
    }
    for line in log.strip().split('\n'):
        line = line.strip().split()
        if not line:
            continue
        if line[-1] == 'ok':
            score += test_scores[line[0]]
    return score


class TestLossAndDerivatives(unittest.TestCase):

    ref_dict = np.load('data_assignment02.npy', allow_pickle=True).item()
    X_ref = ref_dict['X_ref']
    y_ref = ref_dict['y_ref']
    w_hat = ref_dict['w_hat']
    knn_test = LossAndDerivatives

    def test_mse_derivative(self):
        mse_derivative = LossAndDerivatives.mse_derivative(
            self.X_ref, self.y_ref, self.w_hat)
        self.assertTrue(np.allclose(
            mse_derivative, self.ref_dict['mse_derivative'], atol=1e-4))

    def test_mae_derivative(self):
        mae_derivative = LossAndDerivatives.mae_derivative(
            self.X_ref, self.y_ref, self.w_hat)
        self.assertTrue(np.allclose(
            mae_derivative, self.ref_dict['mae_derivative'], atol=1e-4))

    def test_l2_reg_derivative(self):
        l2_reg_derivative = LossAndDerivatives.l2_reg_derivative(self.w_hat)
        self.assertTrue(np.allclose(l2_reg_derivative,
                        self.ref_dict['l2_reg_derivative'], atol=1e-4))

    def test_l1_reg_derivative(self):
        l1_reg_derivative = LossAndDerivatives.l1_reg_derivative(self.w_hat)
        self.assertTrue(np.allclose(l1_reg_derivative,
                        self.ref_dict['l1_reg_derivative'], atol=1e-4))

    def test_mse(self):
        mse = LossAndDerivatives.mse(self.X_ref, self.y_ref, self.w_hat)
        self.assertTrue(np.allclose(mse, self.ref_dict['mse'], atol=1e-4))

    def test_mae(self):
        mae = LossAndDerivatives.mae(self.X_ref, self.y_ref, self.w_hat)
        self.assertTrue(np.allclose(mae, self.ref_dict['mae'], atol=1e-4))

    def test_l2_reg(self):
        l2_reg = LossAndDerivatives.l2_reg(self.w_hat)
        self.assertTrue(np.allclose(
            l2_reg, self.ref_dict['l2_reg'], atol=1e-4))

    def test_l1_reg(self):
        l1_reg = LossAndDerivatives.l1_reg(self.w_hat)
        self.assertTrue(np.allclose(
            l1_reg, self.ref_dict['l1_reg'], atol=1e-4))


suite = unittest.TestLoader().loadTestsFromTestCase(TestSolution)
string_io = io.StringIO()
unittest.TextTestRunner(verbosity=2, stream=string_io).run(suite)
print(string_io.getvalue())
score_basic = get_basic_score(string_io.getvalue())


print(string_io.getvalue(), file=sys.stderr)
print(" basic_score ", score_basic)
