#!/usr/bin/env python
"""
Test stack2image converter
"""
__author__ = "Alex Drlica-Wagner"

import os
import sys
import subprocess
import glob

import numpy as np
import pandas as pd
import pytest
import unittest

sys.path.append('..')

class TestStack2Image(unittest.TestCase):
    def setUp(self):
        self.datadir = "test_data"
        self.filename = os.path.join(self.datadir,"testtile.fits")
        self.outdir = os.path.join(self.datadir,"images")
        pass

    def tearDown(self):
        cmd = 'rm -r {}'.format(self.outdir)
        subprocess.check_output(cmd,shell=True)

    def test_convert_all(self):
        cmd = "../stack2image.py {} -d {} --all".format(self.filename,self.outdir)
        subprocess.check_output(cmd,shell=True)

        assert len(glob.glob(self.outdir+'/*')) == 10

    def test_convert_index(self):
        cmd = "../stack2image.py {} -d {} --index 2 3 5".format(self.filename,self.outdir)
        subprocess.check_output(cmd,shell=True)

        assert len(glob.glob(self.outdir+'/*')) == 3

    def test_convert_cutid(self):
        cmd = "../stack2image.py {} -d {} --cutid 964368035 964420505".format(self.filename,self.outdir)
        subprocess.check_output(cmd,shell=True)

        assert len(glob.glob(self.outdir+'/*')) == 2
