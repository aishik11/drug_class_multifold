import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import utils

def run_test(base_df, df1,df2, ext, seed, khop):
     #read the indications with path
    split_train = df1#pd.read_csv('train_split_8020_43.csv', delimiter=",") # read the train split
    split_test = df2#pd.read_csv('test_split_8020_43.csv', delimiter=",") # read the test split

    merged_df_train = pd.merge(split_train, base_df,  how='left', left_on=['drug','disease'], right_on = ['drug','disease']) # merge the split with the path
    merged_df_test = pd.merge(split_test, base_df,  how='left', left_on=['drug','disease'], right_on = ['drug','disease'])# merge the split with the path



    # we first work on the train
    df = merged_df_train
    df = df[df['path_it'].notna()]
    df['path_it'] = df['path_it'].apply(literal_eval)

    data = []
    for di in df['disease'].unique():
      # for each disease calculate the gene info
      data = data + utils.get_int_map_old(df, di, khop)


    di_gene_df = pd.DataFrame(data, columns=['disease', 'gene', 'dm', 'sym', 'not'])
    #di_gene_df.to_csv('di_gene_df50.csv')

    #known_genes  = di_gene_df['gene'].unique()


    def avg_prob(x):
        e_x = x
        return e_x / e_x.sum(axis=1, keepdims=True)

    def score_path(path, known_genes, map):
        #return sum(map[gene[0]] for gene in path if gene[0] in known_genes)
        return sum(map[gene[0]] if gene[0] in known_genes else -1 for gene in path[1:-1])

    # we work with the test split
    #merged_df_test
    df = merged_df_test
    df = df[df['path_it'].notna()]
    df['path_it'] = df['path_it'].apply(literal_eval)


    path_vec = []

    similarity = pd.read_csv('similarity.csv')

    unknown = 0
    known = 0
    fixed = 0
    for i, r in tqdm(df.iterrows(),total=len(df)):

        ints = [] # store the list of genes
        map_dm = {} # store the count of gene
        paths = r['path_it']

        sub_df = di_gene_df[di_gene_df['disease'] == r['disease']]
        known_genes = sub_df['gene'].unique()

        sub_df_sim = similarity[similarity['node1'].isin(known_genes)]
        probs = sub_df[['dm', 'sym', 'not']].values
        softmax_probs = avg_prob(probs)

        # Add the new columns back to the DataFrame
        sub_df['dm_a'] = softmax_probs[:, 0]
        sub_df['sym_a'] = softmax_probs[:, 1]
        sub_df['not_a'] = softmax_probs[:, 2]
        sub_df['max'] = sub_df[['dm_a','sym_a','not_a']].max(axis=1)

        allowed_known = []#sub_df[sub_df['max'] >= 0.5]['gene'].unique()

        allowed_known_map = {}

        for i, sr in sub_df.iterrows():
            if sr['max'] >= 0.4:
                allowed_known.append(sr['gene'])
                allowed_known_map[sr['gene']] = sr[['dm','sym','not']].to_list()[np.argmax(sr[['dm_a','sym_a','not_a']])]
            else:
                allowed_known_map[sr['gene']] = 0
        for p in paths[:]:

          for n in p:

            if n[0] in map_dm.keys():
              map_dm[n[0]]+=1
            else:
              map_dm[n[0]] = 1

        for k in map_dm.keys():
            map_dm[k] = 1#map_dm[k]/len(paths)


        scored_paths = [(score_path(path, allowed_known, allowed_known_map), path) for path in paths]
        sorted_paths = sorted(scored_paths, key=lambda x: x[0], reverse=True)
        sorted_paths_only = [path for score, path in sorted_paths]

        for p in sorted_paths_only[:200]:
          #unknown+=sum(gene[0] not in known_genes for gene in p)
          #known += sum(gene[0] in known_genes for gene in p)
          for n in p[1:-1]:
            '''
            if n[0] in map_dm.keys():
              map_dm[n[0]]+=1
            else:
              map_dm[n[0]] = 1
            '''
            if n[1] == 'Gene':
                ints.append(n[0])
          ints = list(set(ints))

        for i in ints:

          #row = di_gene_df[di_gene_df['disease'] == r['disease']][di_gene_df['gene'] == i]
          row = sub_df[sub_df['gene'] == i]
          #print(row)
          if not row.empty:
            known += 1
            path_vec.append([r['disease'],r['drug'],i,'NA','NA',map_dm[i], row['dm'].values[0], row['sym'].values[0], row['not'].values[0], r['category']])
          else:
            unknown+=1
            matching_sim = sub_df_sim[sub_df_sim['node2']==i]

            if not matching_sim.empty:
                fixed+=1
                row = di_gene_df[di_gene_df['disease'] == r['disease']][di_gene_df['gene'] == matching_sim['node1'].values[0]]
                path_vec.append([r['disease'],r['drug'],i,matching_sim['node1'].values[0],matching_sim['similarity'].values[0],map_dm[i], row['dm'].values[0], row['sym'].values[0], row['not'].values[0], r['category']])
            else:
                path_vec.append([r['disease'],r['drug'],i,'NA','NA',map_dm[i], 'NA', 'NA','NA', r['category']])


    test_df = pd.DataFrame(path_vec, columns=['disease','drug', 'gene','gene_replaced_by','similarity','occurance_across_path', 'dm', 'sym', 'not','actual_category'])
    test_df.to_csv(f'test_df{khop}_'+ext+'_' + seed + ".csv")

    #utils.get_accuracy(test_df) # calculate the accuracy
    utils.get_accuracy_avg(test_df)
    test_df.to_csv(f'test_df{khop}_evaluated_'+ext+'_' + seed + ".csv")
    print('average unknown',unknown/len(df))
    print('average known',known/len(df))
    print('average fixed',fixed/len(df))

    return test_df


if __name__ == "__main__":
    split_train = pd.read_csv('split_train.csv', delimiter=",") # read the train split
    split_test = pd.read_csv('split_test.csv', delimiter=",") # read the test split
    base_df = pd.read_csv('indications_500.csv', delimiter=",")[['drug', 'disease', 'path_it']]
    df = pd.concat([split_train, split_test])


    splits = [0.2,0.3]

    seed = 42

    khops = [50,100,150,200,250,300,350,400,450,500] # max can be 500
    for split in splits:
        for khop in khops:
            for i in range(1):
                seed = 41 + i
                train_df, test_df = train_test_split(df, test_size=split, random_state=seed)

                train_df.to_csv(f'train_split_{str(split)}_' + str(seed) + ".csv", index=False)
                test_df.to_csv(f'test_split_{str(split)}_' + str(seed) + ".csv", index=False)

                testdf = run_test(base_df, train_df, test_df, str(split), str(seed), khop)

