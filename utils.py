import numpy as np
import pandas as pd

def get_int_map_old(df, disease, khop):
  map_dm = {}
  np = 0
  for paths in df[df['disease'] == disease][df['category']=='DM']['path_it']:
    for p in paths[:khop]:
      np +=1
      for n in p:
        if n[1] == 'Disease' or n[1] == 'Compound':
          continue
        if n[0] in map_dm.keys():
          map_dm[n[0]]+=1
        else:
          map_dm[n[0]] = 1

  for k in map_dm.keys():
    map_dm[k] = map_dm[k]/np

  map_sym = {}
  np = 0
  for paths in df[df['disease'] == disease][df['category']=='SYM']['path_it']:
    for p in paths[:khop]:
      np +=1
      for n in p:
        if n[1] == 'Disease' or n[1] == 'Compound':
          continue
        if n[0] in map_sym.keys():
          map_sym[n[0]]+=1
        else:
          map_sym[n[0]] = 1
  for k in map_sym.keys():
    map_sym[k] = map_sym[k]/np


  map_not = {}
  np = 0
  for paths in df[df['disease'] == disease][df['category']=='NOT']['path_it']:
    for p in paths[:khop]:
      np +=1
      for n in p:
        if n[1] == 'Disease' or n[1] == 'Compound':
          continue
        if n[0] in map_not.keys():
          map_not[n[0]]+=1
        else:
          map_not[n[0]] = 1
  for k in map_not.keys():
    map_not[k] = map_not[k]/np

  all_int = list(set(list(map_dm.keys())  + list(map_sym.keys()) + list(map_not.keys())))

  final_list = []

  for i in all_int:
    dm = 0
    sym = 0
    npt = 0
    if i in map_dm.keys():
      dm = map_dm[i]
    if i in map_sym.keys():
      sym = map_sym[i]
    if i in map_not.keys():
      npt = map_not[i]

    final_list.append([disease,i, dm, sym, npt])
  return final_list



def get_int_map(df, disease, khop):
    map_dm = {} #map of gene vs average
    map_sym = {}
    map_not = {}

    sdf = df[df['disease'] == disease] # dataframe only with this disease data
    drugs = sdf['drug'].unique() # all drugs for this disease

    for d in drugs:# calculate the average for each data

        map_dm_d = {} #map of gene vs average for this drug
        map_sym_d = {}
        map_not_d = {}
        np = 0
        for paths in sdf[sdf['drug'] == d][sdf['category'] == 'DM']['path_it']:

            for p in paths[:khop]:
                np += 1
                for n in p:
                    if n[1] == 'Disease' or n[1] == 'Compound': #ignore disease, compound
                        continue
                    if n[0] in map_dm_d.keys():
                        map_dm_d[n[0]] += 1
                    else:
                        map_dm_d[n[0]] = 1

        for k in map_dm_d.keys(): # calculate the drug level average
            map_dm_d[k] = map_dm_d[k] / np
            if k in map_dm.keys():
                map_dm[k] += map_dm_d[k] # add to the disease level averaeg. we will further average this later
            else:
                map_dm[k] = map_dm_d[k]

        np = 0
        for paths in sdf[sdf['drug'] == d][sdf['category'] == 'SYM']['path_it']:
            for p in paths[:khop]:
                np += 1
                for n in p:
                    if n[1] == 'Disease' or n[1] == 'Compound':
                        continue
                    if n[0] in map_sym_d.keys():
                        map_sym_d[n[0]] += 1
                    else:
                        map_sym_d[n[0]] = 1
        for k in map_sym_d.keys():
            map_sym_d[k] = map_sym_d[k] / np
            if k in map_sym.keys():
                map_sym[k] += map_sym_d[k]
            else:
                map_sym[k] = map_sym_d[k]

        np = 0
        for paths in sdf[sdf['drug'] == d][sdf['category'] == 'NOT']['path_it']:
            for p in paths[:khop]:
                np += 1
                for n in p:
                    if n[1] == 'Disease' or n[1] == 'Compound':
                        continue
                    if n[0] in map_not_d.keys():
                        map_not_d[n[0]] += 1
                    else:
                        map_not_d[n[0]] = 1
        for k in map_not_d.keys():
            map_not_d[k] = map_not_d[k] / np
            if k in map_not.keys():
                map_not[k] += map_not_d[k]
            else:
                map_not[k] = map_not_d[k]

    all_int = list(set(list(map_dm.keys()) + list(map_sym.keys()) + list(map_not.keys())))

    final_list = []

    for i in all_int:
        dm = 0
        sym = 0
        npt = 0
        if i in map_dm.keys():
            dm = map_dm[i] / len(drugs) # for averaget the sum of all drug gene freq
        if i in map_sym.keys():
            sym = map_sym[i] / len(drugs)
        if i in map_not.keys():
            npt = map_not[i] / len(drugs)

        final_list.append([disease, i, dm, sym, npt])
    return final_list


def get_accuracy_avg(test_df):
    diseases = test_df['disease'].unique()

    cat = ['DM', 'SYM', 'NOT']

    correct = 0
    dm_cor = 0
    sym_cor = 0
    not_cor = 0

    dm_t = 0
    sym_t = 0
    not_t = 0
    for di in diseases:
        di_df = test_df[test_df['disease'] == di]

        drugs = di_df['drug'].unique()
        for d in drugs:
            dr_df = di_df[di_df['drug'] == d]

            dm = 0
            sym = 0
            not_ = 0

            total_comp = [dr_df['dm'].replace('NA', 0).sum()/len(dr_df), dr_df['sym'].replace('NA', 0).sum()/len(dr_df),dr_df['not'].replace('NA', 0).sum()/len(dr_df)]
            r = dr_df.iloc[0]
            '''
            for i, r in dr_df.iterrows():
                comp = r[['dm', 'sym', 'not']].replace('NA', 0)
                #comp = np.absolute(np.array(comp) - r['occurance_across_path'] / 50)
                #index_min = np.argmin(comp)
                index_min = np.argmax(comp)
                if index_min == 0:
                    dm += 1
                elif index_min == 1:
                    sym += 1
                else:
                    not_ += 1
            '''
            if r['actual_category'] == 'DM':
                dm_t += 1

            if r['actual_category'] == 'SYM':
                sym_t += 1

            if r['actual_category'] == 'NOT':
                not_t += 1

            if cat[np.argmax(total_comp)] == r['actual_category']:
                correct += 1
                if r['actual_category'] == 'DM':
                    dm_cor += 1

                if r['actual_category'] == 'SYM':
                    sym_cor += 1
                    print(r)

                if r['actual_category'] == 'NOT':
                    not_cor += 1

            for i,ro in dr_df.iterrows():
                test_df.at[i,'prediction'] = cat[np.argmax(total_comp)]
                test_df.at[i, 'result'] = (cat[np.argmax(total_comp)] == r['actual_category'])

                    

            # print(di, d, cat[np.argmax([dm, sym, not_])], r['actual_category'])

    print('Total accuracy ',correct / (dm_t + sym_t + not_t))
    print('DM accuracy ',dm_cor / dm_t)
    print('SYM accuracy ',sym_cor / sym_t)
    #print('NOT accuracy ',not_cor / not_t)

def get_accuracy(test_df):
    diseases = test_df['disease'].unique()

    cat = ['DM', 'SYM', 'NOT']

    correct = 0
    dm_cor = 0
    sym_cor = 0
    not_cor = 0

    dm_t = 0
    sym_t = 0
    not_t = 0
    for di in diseases:
        di_df = test_df[test_df['disease'] == di]

        drugs = di_df['drug'].unique()
        for d in drugs:
            dr_df = di_df[di_df['drug'] == d]

            dm = 0
            sym = 0
            not_ = 0

            #total_comp = [dr_df['dm'].sum(), dr_df['sym'].sum(),dr_df['not'].sum()]
            #r = total_comp.iloc[0]

            for i, r in dr_df.iterrows():
                comp = r[['DM', 'SYM', 'NOT']].replace('NA', 0)
                #comp = np.absolute(np.array(comp) - r['occurance_across_path'] / 50)
                #index_min = np.argmin(comp)
                index_min = np.argmax(comp)
                if index_min == 0:
                    dm += 1
                elif index_min == 1:
                    sym += 1
                else:
                    not_ += 1

            if r['actual_category'] == 'DM':
                dm_t += 1

            if r['actual_category'] == 'SYM':
                sym_t += 1

            if r['actual_category'] == 'NOT':
                not_t += 1

            if cat[np.argmax([dm, sym, not_])] == r['actual_category']:
                correct += 1
                if r['actual_category'] == 'DM':
                    dm_cor += 1

                if r['actual_category'] == 'SYM':
                    sym_cor += 1

                if r['actual_category'] == 'NOT':
                    not_cor += 1

            # print(di, d, cat[np.argmax([dm, sym, not_])], r['actual_category'])

    print('Total accuracy ',correct / (dm_t + sym_t + not_t))
    print('DM accuracy ',dm_cor / dm_t)
    print('SYM accuracy ',sym_cor / sym_t)
    print('NOT accuracy ',not_cor / not_t)