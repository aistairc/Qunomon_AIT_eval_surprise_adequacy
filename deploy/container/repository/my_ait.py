#!/usr/bin/env python
# coding: utf-8

# # AIT Development notebook

# ## notebook of structure

# | #  | Name                                               | cells | for_dev | edit               | description                                                                |
# |----|----------------------------------------------------|-------|---------|--------------------|----------------------------------------------------------------------------|
# | 1  | [Environment detection](##1-Environment-detection) | 1     | No      | uneditable         | detect whether the notebook are invoked for packaging or in production     |
# | 2  | [Preparing AIT SDK](##2-Preparing-AIT-SDK)         | 1     | Yes     | uneditable         | download and install AIT SDK                                               |
# | 3  | [Dependency Management](##3-Dependency-Management) | 3     | Yes     | required(cell #2)  | generate requirements.txt for Docker container                             |
# | 4  | [Importing Libraries](##4-Importing-Libraries)     | 2     | Yes     | required(cell #1)  | import required libraries                                                  |
# | 5  | [Manifest Generation](##5-Manifest-Generation)     | 1     | Yes     | required           | generate AIT Manifest                                                      |
# | 6  | [Prepare for the Input](##6-Prepare-for-the-Input) | 1     | Yes     | required           | generate AIT Input JSON (inventory mapper)                                 |
# | 7  | [Initialization](##7-Initialization)               | 1     | No      | uneditable         | initialization for AIT execution                                           |
# | 8  | [Function definitions](##8-Function-definitions)   | N     | No      | required           | define functions invoked from Main area.<br> also define output functions. |
# | 9  | [Main Algorithms](##9-Main-Algorithms)             | 1     | No      | required           | area for main algorithms of an AIT                                         |
# | 10 | [Entry point](##10-Entry-point)                    | 1     | No      | uneditable         | an entry point where Qunomon invoke this AIT from here                     |
# | 11 | [License](##11-License)                            | 1     | Yes     | required           | generate license information                                               |
# | 12 | [Deployment](##12-Deployment)                      | 1     | Yes     | uneditable         | convert this notebook to the python file for packaging purpose             |

# ## notebook template revision history

# 1.0.1 2020/10/21
# 
# * add revision history
# * separate `create requirements and pip install` editable and noeditable
# * separate `import` editable and noeditable
# 
# 1.0.0 2020/10/12
# 
# * new cerarion

# ## body

# ### #1 Environment detection

# [uneditable]

# In[1]:


# Determine whether to start AIT or jupyter by startup argument
import sys
is_ait_launch = (len(sys.argv) == 2)


# ### #2 Preparing AIT SDK

# [uneditable]

# In[2]:


if not is_ait_launch:
    # get ait-sdk file name
    from pathlib import Path
    from glob import glob
    import re
    import os

    current_dir = get_ipython().run_line_magic('pwd', '')

    ait_sdk_path = "./ait_sdk-*-py3-none-any.whl"
    ait_sdk_list = glob(ait_sdk_path)
    ait_sdk_name = os.path.basename(ait_sdk_list[-1])

    # install ait-sdk
    get_ipython().system('pip install -q --upgrade pip')
    get_ipython().system('pip install -q --no-deps --force-reinstall ./$ait_sdk_name')


# ### #3 Dependency Management

# #### #3-1 [uneditable]

# In[3]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_requirements_generator import AITRequirementsGenerator
    requirements_generator = AITRequirementsGenerator()


# #### #3-2 [required]

# In[4]:


if not is_ait_launch:
    requirements_generator.add_package('numpy', '1.26.4')
    requirements_generator.add_package('matplotlib', '3.9.4')
    requirements_generator.add_package('pandas', '2.2.3')
    requirements_generator.add_package('scikit-learn', '1.6.0')
    requirements_generator.add_package('tensorflow', '2.15.1')
    requirements_generator.add_package('tqdm', '4.67.1')


# #### #3-3 [uneditable]

# In[5]:


if not is_ait_launch:
    requirements_generator.add_package(f'./{ait_sdk_name}')
    requirements_path = requirements_generator.create_requirements(current_dir)

    get_ipython().system('pip install -q -r $requirements_path ')


# ### #4 Importing Libraries

# #### #4-1 [required]

# In[6]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from scipy.stats import norm

from collections import defaultdict 
import random
import numpy as np
from sklearn.utils import shuffle
from scipy import stats
from sklearn.metrics import precision_score, recall_score, accuracy_score 
from collections import Counter

import pandas as pd

from tqdm import tqdm


# #### #4-2 [uneditable]

# In[7]:


# must use modules
from os import path
import shutil  # do not remove
from ait_sdk.common.files.ait_input import AITInput  # do not remove
from ait_sdk.common.files.ait_output import AITOutput  # do not remove
from ait_sdk.common.files.ait_manifest import AITManifest  # do not remove
from ait_sdk.develop.ait_path_helper import AITPathHelper  # do not remove
from ait_sdk.utils.logging import get_logger, log, get_log_path  # do not remove
from ait_sdk.develop.annotation import measures, resources, downloads, ait_main  # do not remove
# must use modules


# ### #5 Manifest Generation

# [required]

# In[8]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_manifest_generator import AITManifestGenerator
    manifest_generator = AITManifestGenerator(current_dir)
    manifest_generator.set_ait_name('eval_surprise_adequacy')
    manifest_generator.set_ait_description('入力VAEモデルのサプライズ適切性（SA）を計算しています。SAは、入力データの各サンプルに対する各ニューロンの活動トレースを評価します。詳細については、元の論文「Kim, et al. Evaluating Surprise Adequacy for Deep Learning System Testing」（URL: https://dl.acm.org/doi/full/10.1145/3546947）')
    manifest_generator.set_ait_source_repository('https://github.com/aistairc/Qunomon_AIT_eval_surprise_adequacy')
    manifest_generator.set_ait_version('1.3')
    manifest_generator.add_ait_licenses('Apache License Version 2.0')
    manifest_generator.add_ait_keywords('evaluation')
    manifest_generator.set_ait_quality('https://ait-hub.pj.aist.go.jp/ait-hub/api/0.0.1/qualityDimensions/機械学習品質マネジメントガイドライン第三版/C-1機械学習モデルの正確性')
    
    # inventory
    inventory_requirement_image_dataset = manifest_generator.format_ait_inventory_requirement(format_=['npz'])
    manifest_generator.add_ait_inventories(name='image_dataset', 
                                           type_='dataset', 
                                           description='画像データセット ※以下の4つのみ利用可能：mnist data, fashion mnist data, cifar10 data, cifar100 data', 
                                           requirement=inventory_requirement_image_dataset)
    
    inventory_requirement_vae_model = manifest_generator.format_ait_inventory_requirement(format_=['keras'])
    manifest_generator.add_ait_inventories(name='vae', 
                                           type_='model', 
                                           description='学習済みのVAEモデル', 
                                           requirement=inventory_requirement_vae_model)
    
    # input parameters, Hyperparameters
    manifest_generator.add_ait_parameters(name='latent_dim', 
                                          type_='int', 
                                          default_val='100', 
                                          description='潜在空間の次元を指定するハイパーパラメーター')

    # output
    manifest_generator.add_ait_downloads(name='DSA', description='Data Structure Analysis of given data with given model')
    manifest_generator.add_ait_downloads(name='Log', description='AIT実行ログ')
    manifest_path = manifest_generator.write()


# ### #6 Prepare for the Input

# [required]

# In[9]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_input_generator import AITInputGenerator
    input_generator = AITInputGenerator(manifest_path)
    input_generator.add_ait_inventories(name='image_dataset',
                                        value='mnist_data/mnist_train_data.npz')
    input_generator.add_ait_inventories(name='vae', 
                                        value='vae_model/vae_mnist_Sys_10.keras')
    
    latent_dim = 100
    input_generator.set_ait_params(name='latent_dim', value=latent_dim)

    input_generator.write()


# ### #7 Initialization

# [uneditable]

# In[10]:


logger = get_logger()

ait_manifest = AITManifest()
ait_input = AITInput(ait_manifest)
ait_output = AITOutput(ait_manifest)

if is_ait_launch:
    # launch from AIT
    current_dir = path.dirname(path.abspath(__file__))
    path_helper = AITPathHelper(argv=sys.argv, ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)
else:
    # launch from jupyter notebook
    # ait.input.json make in input_dir
    input_dir = '/usr/local/qai/mnt/ip/job_args/1/1'
    current_dir = get_ipython().run_line_magic('pwd', '')
    path_helper = AITPathHelper(argv=['', input_dir], ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)

ait_input.read_json(path_helper.get_input_file_path())
ait_manifest.read_json(path_helper.get_manifest_file_path())

### do not edit cell


# ### #8 Function definitions

# [required]

# In[11]:


@log(logger)
@downloads(ait_output, path_helper, 'DSA', 'DSA_values.csv')
def calculate_DSA(vae_model, train_data, file_path: str = None):
    encoder = vae_model.get_layer('Encoder')
    decoder = vae_model.get_layer('Decoder')
    classifier = vae_model.get_layer('Classifier')

    print('Encoder prediction ...')
    train_activations = encoder.predict(train_data)[2]
    print('Classifier prediction ...')
    train_predictions = classifier.predict(train_activations)
    train_predictions = np.argmax(train_predictions, axis=1)
    valid_indices = np.logical_and(np.isfinite(train_activations).all(axis=1),
                                   np.isfinite(train_predictions))
    train_activations = train_activations[valid_indices]
    train_predictions = train_predictions[valid_indices]

    def _DSA(X, y):
        n_samples = X.shape[0]
        dsa_scores = np.zeros(n_samples)

        for i in tqdm(range(n_samples), desc='Calculating DSA'):
            distances = np.linalg.norm(X - X[i], axis=1)
            same_class_mask = (y == y[i])
            diff_class_mask = (y != y[i])

            same_class_distances = distances[same_class_mask]
            diff_class_distances = distances[diff_class_mask]

            min_same_class_distance = np.min(same_class_distances[same_class_distances > 0])
            min_diff_class_distance = np.min(diff_class_distances)

            dsa_scores[i] = min_same_class_distance / min_diff_class_distance

        return dsa_scores

    surprise_adequacy_dsa = _DSA(train_activations, train_predictions)
    pd.DataFrame(surprise_adequacy_dsa).to_csv(file_path)
    return surprise_adequacy_dsa


# In[12]:


@log(logger)
@downloads(ait_output, path_helper, 'Log', 'ait.log')
def move_log(file_path: str=None) -> str:
    shutil.move(get_log_path(), file_path)


# ### #9 Main Algorithms

# [required]

# In[13]:


# parameter
latent_dim = ait_input.get_method_param_value('latent_dim')

@log(logger)
@ait_main(ait_output, path_helper)
def main() -> None:
    
    # inventories
    input_data = np.load(ait_input.get_inventory_path('image_dataset'))
    
    model = keras.models.load_model(ait_input.get_inventory_path('vae'), compile=False)

    dsa = calculate_DSA(model, input_data['X'])
    
    move_log()


# ### #10 Entry point

# [uneditable]

# In[ ]:


if __name__ == '__main__':
    main()


# ### #11 License

# [required]

# In[ ]:


ait_owner='AIST'
ait_creation_year='2024'


# ### #12 Deployment

# [uneditable] 

# In[ ]:


if not is_ait_launch:
    from ait_sdk.deploy import prepare_deploy
    from ait_sdk.license.license_generator import LicenseGenerator
    
    current_dir = get_ipython().run_line_magic('pwd', '')
    prepare_deploy(ait_sdk_name, current_dir, requirements_path)
    
    # output License.txt
    license_generator = LicenseGenerator()
    license_generator.write('../top_dir/LICENSE.txt', ait_creation_year, ait_owner)


# In[ ]:




