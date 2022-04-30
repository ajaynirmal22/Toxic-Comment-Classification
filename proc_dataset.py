from utils import *


class prepare_data():

    def __init__(self, p_filepath):
        self.m_classes = ['toxicity','severe_toxicity','obscene','threat','insult','sexual_explicit']

        self.data_proc = self._init_df(p_filepath)


    def _init_df(self, p_filepath):
        df_procData = pd.read_csv(p_filepath)
        return df_procData
    

    def _create_proc_dataset(self):
        df_0 = self.data_proc.drop(columns=["Unnamed: 0"],axis=1).dropna(axis=0)
        
        df_t = df_0.drop(columns=['comment_text'])
        
        df_t['severe_toxicity'] = np.where(df_0['severe_toxicity'] < 0.4, 2.5*df_0['severe_toxicity'],df_0['severe_toxicity']) 
        
        df_t = pd.concat([df_t[df_t['toxicity'] >= 0.5].sample(frac = 3, replace=True), df_t[df_t['toxicity'] < 0.5]])

        bins = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        labels = [0,1,2,3,4,5,6,7,8,9]
        for i in ['severe_toxicity','obscene','sexual_explicit','threat']:
            df_t[i] = pd.cut(df_t[i], bins=bins, labels=labels, include_lowest=True)
        
        df_t['toxicity'] = np.where(df_t['toxicity'] < 0.5, 0, 1) 
        df_t['insult'] = np.where(df_t['insult'] < 0.1, 0, 1) 
        
        df_t = df_t.reset_index(drop=True)

        pure_indices = df_t[(df_t['toxicity'] == 0) & 
            (df_t['severe_toxicity'] == 0) & 
            (df_t['obscene'] == 0) & 
            (df_t['sexual_explicit'] == 0) & 
            (df_t['insult'] == 0) &
            (df_t['threat'] == 0)
            ].index
        
        df_tt = df_t.drop(pure_indices,axis=0)
        return df_tt
    

    def _create_datasets(self, p_df_tt):
        frac = [0.9,0.15,0.2,0.2,1.0,0.05]
        datasets = dict()

        for i,column in enumerate(self.m_classes):
            if i==4:
                df_c1 = p_df_tt.drop(columns=p_df_tt.columns.difference([column,'preprocessed_text']))
                temp = df_c1[df_c1[column] == 1].sample(frac = 0.05)
                df_c1.drop(df_c1[df_c1[column] == 1].index,axis=0,inplace=True)
                df_c1 = pd.concat([temp,df_c1],axis=0).reset_index(drop=True)
                datasets[column] = df_c1
            else:
                df_c1 = p_df_tt.drop(columns=p_df_tt.columns.difference([column,'preprocessed_text']))
                temp = df_c1[df_c1[column] == 0].sample(frac = frac[i])
                df_c1.drop(df_c1[df_c1[column] == 0].index,axis=0,inplace=True)
                df_c1 = pd.concat([temp,df_c1],axis=0).reset_index(drop=True)
                datasets[column] = df_c1
        
        return datasets

    
    def fit(self):
        r_proc_dataset = self._create_proc_dataset()
        r_datasets = self._create_datasets(r_proc_dataset)

        return r_datasets, r_proc_dataset