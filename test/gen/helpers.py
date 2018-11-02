#!python
#
# Copyright 2018 Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import string

import numpy as np
import tensorflow as tf

LICENSE = r"""/*
 * Copyright 2018 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */"""


def get_license():
    return LICENSE


COMMENT_TPL = r"""
// DO NOT MODIFY BY HAND
// This file was automatically generated by {scriptname}.
// Results calculated using Tensorflow v{tf_version}."""


def get_dont_modify_comment(scriptname):
    tf_version = tf.VERSION
    return COMMENT_TPL.format(scriptname=scriptname, tf_version=tf_version)


# To ensure that the tests can be computed exactly, we require that the results
# are contained in the set of values that can be exactly represented by the
# floating point data types. Single precision floating point values can
# represent all integers up to 2^24. If the values would be larger than this,
# then limit the input values to a smaller range of possible values to limit
# the size of the output values.
REQUIRED_MAX = 2**24


def get_result_and_size(func, max_input_val=2**12, floor_div=False, **kwargs):
    """
    Compute the result of func called with passed keyword arguments, ensuring
    that the resulting values are less than the REQUIRED_MAX, and if not adjust
    the maximum values in the input tensors, with those values being set by 
    max_input_val.

    floor_div is used in the case of max_input_val taking a value other than
    a power of 2. This ensures that inputs remain consistent for other functions
    regardless of the value of max_input_val.
    """
    max_output_val = REQUIRED_MAX + 1
    while max_output_val > REQUIRED_MAX:
        if floor_div:
            max_input_val = max_input_val // 2
        else:
            max_input_val /= 2
        output = func(max_input_val, **kwargs)
        max_output_val = np.max(output)
    return output, max_input_val


def get_tensor_data(size, max_val):
    "Get a list of data values to use as input data."
    if max_val < 1:
        return [i for i in range(1, size + 1)]
    else:
        return [(i % max_val) + 1 for i in range(size)]


def get_signed_tensor_data(size, min_val, max_val):
    """
    As with get_tensor_data(), but enables negative-valued data
    as well.
    """
    assert(min_val < max_val)
    res = []

    while len(res) < size:
        res.extend([i for i in range(min_val, max_val)])

    return res[0:size]


SPACE_REGEX = re.compile(r'([0-9]\.?)\s+')


def format_tensor(tensor):
    "Convert a numpy tensor into an initializer list."
    t_str = np.array2string(
        tensor.flatten(), floatmode='unique', separator=', ')
    t_braced = '{' + t_str[1:-1] + '}'
    return t_braced


def to_lower_case_str(var):
    "Convert var to a string of lower case chars."
    return str(var).lower()


def to_camel_case(snake_case):
    "Convert a snake_case string to CamelCase."
    return string.capwords(snake_case, '_').replace('_', '')


def get_test_directory():
    "Get the root test directory for SYCL-DNN."
    file_path = os.path.dirname(__file__)
    test_dir = os.path.normpath(os.path.join(file_path, ".."))
    return test_dir
