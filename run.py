from utils import *
from train import *
from validation import *
from test import *
from proc_pipeline import *
from proc_dataset import *
from save_model import save_models

proc_pipeline_obj = preprocessing()

prep_dataset_obj = prepare_data('dataset/all_data.csv')
r_datasets, r_proc_dataset = prep_dataset_obj.transform()

train = train_models(r_datasets,r_proc_dataset)
trained_models = train.fit()

save_models = save_models(trained_models,train.m_vectorizer)
save_models.save()