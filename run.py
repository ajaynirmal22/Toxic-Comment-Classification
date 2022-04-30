from utils import *
from train import *
from validation import *
from test import *
from proc_pipeline import *
from proc_dataset import *
from save_model import save_models

print("reading the dataset...")
df = pd.read_csv("datasets/all_data.csv")
proc_pipeline_obj = preprocessing(df)
print("preprocessing the dataset...")
df_proc = proc_pipeline_obj.fit_transform()
print("preparing the train dataset...")
prep_dataset_obj = prepare_data(df_proc)
r_datasets, r_proc_dataset = prep_dataset_obj.transform()
print("starting the training...")
train = train_models(r_datasets,r_proc_dataset)
trained_models = train.fit()
print("saving the models...")
save_models = save_models(trained_models,train.m_vectorizer)
save_models.save()