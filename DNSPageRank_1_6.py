# PAGERANK Algorithm 1.6
# Improvement over v1.5 including:
# - Sinkholes go to 0
# - Correcting indices of data.
# - Using a single Link Matrix
# - Making reputation vectors sparse
# - Added memory profiling
# - Redirected output into log file

import pandas as pd
import numpy as np
import pickle
import time
import os
import sys


from scipy import sparse

sys.stdout = open('pagerank.log', 'w')


def load_reputations_from_files():
    ips_file = os.path.join('/work/users/fabiomz','pagerank_ips.parquet')
    ips = pd.read_parquet(ips_file,columns=['ip_id', 'reputation'])
    #ips = ips.astype(np.int32)
    domains_file = os.path.join('/work/users/fabiomz','pagerank_domains.parquet')
    domains = pd.read_parquet(domains_file,columns=['domain_id', 'reputation'])
    #domains = domains.astype(np.int32)
    return ips, domains

def load_links_from_files():
    records_file = os.path.join('/work/users/fabiomz','pagerank_records.parquet')
    records = pd.read_parquet(records_file,columns=['ip_id', 'domain_id'])
    #records = records.astype(np.int32)
    return records



def load_test_reputations():
    ips_data = {'ip_id': np.array([0,1,2,3])+1,  
                'reputation': [0, 1, 2, 3]}
    ips = pd.DataFrame(ips_data, dtype=np.int32)
    
    domains_data = {'domain_id': np.array([0,1,2])+1,
                    'reputation': [1, 1, 2]}
    domains = pd.DataFrame(domains_data, dtype=np.int32)
    
    return ips, domains

def load_test_links():    
    records_data = {'domain_id': np.array([0.0,1,1,2,1,2])+1, 
                    'ip_id': np.array([0,0,1,2,2,3])+1,
                    'reputation': [1,1,1,0,0,0]}
    records = pd.DataFrame(records_data, dtype=np.int32)
    
    return records



def update_reputation(ips, domains):
    domains.loc[domains['reputation']==3,'reputation'] = -1
    domains.loc[domains['reputation']==2,'reputation'] = -1
    
    ips.loc[ips['reputation']==3,'reputation'] = -1
    ips.loc[ips['reputation']==2,'reputation'] = -1
    
    return ips,domains

def update_ids(records):
    records['ip_id'] = records['ip_id'] - 1
    records['domain_id'] = records['domain_id'] - 1
    
    return records

def writefile(obj,filename):
    fd = open(filename+'_'+str(int(time.time()))+'.pkl','wb')
    pickle.dump(obj, fd)
    fd.close()
    



print ('{0} - Starting'.format(time.strftime('%X %x')))

#Load reputations dataframes
#ips, domains = load_reputations_from_files()
ips, domains = load_test_reputations()
print ('{0} - Loaded reputation dataframes'.format(time.strftime('%X %x')))

#Reset reputation to be meaningful
ips, domains = update_reputation(ips, domains)
print ('{0} - Updated reputations'.format(time.strftime('%X %x')))

# Get dimensions of domains and ips
D_domains = domains.shape[0]
D_ips = ips.shape[0]

# Set the arrays containing the reputation
rep_sent_domains = sparse.csc_matrix(np.array(domains['reputation'],dtype=np.float16).reshape(D_domains,1))
rep_sent_ips = sparse.csc_matrix(np.array(ips['reputation'],dtype=np.float16).reshape(D_ips,1))

rep_received_domains = sparse.csr_matrix(np.zeros(D_domains,dtype=np.float16).reshape(D_domains,1))
rep_received_ips = sparse.csr_matrix(np.zeros(D_ips,dtype=np.float16).reshape(D_ips,1))
print ('{0} - Instantiated sparse reputation vectors'.format(time.strftime('%X %x')))

# Drop reputation dataframe
del domains
del ips
print ('{0} - Dropped reputation dataframes'.format(time.strftime('%X %x')))



# Load links dataframes
#records = load_links_from_files()
records = load_test_links()
print ('{0} - Loaded link dataframe'.format(time.strftime('%X %x')))

#Rescale id to match the array numbering
records = update_ids(records)
print ('{0} - Updated ids'.format(time.strftime('%X %x')))

# Instantiate sparse matrices
vals = np.ones(records.shape[0], dtype=np.bool_)
linkM = sparse.coo_matrix((vals, (records['domain_id'].tolist(),records['ip_id'].tolist())), shape=(D_domains,D_ips))
print ('{0} - Instantiated and populated sparse link matrices'.format(time.strftime('%X %x')))

# Drop link dataframe
del records
print('{0} - Dropped link dataframe'.format(time.strftime('%X %x')))

#Set the params for the algorithm
alpha_dump = 1
change = np.inf; changes = []; change_threshold = 1
iteration = 0; max_iter=np.inf
print('{0} - PageRank with dimensions (domains x ips) {1} x {2} set'.format(time.strftime('%X %x'),D_domains,D_ips))


while (change > change_threshold and iteration < max_iter):
    
    # Update the matrix of reputation cumulatively
    M1_domains_to_ips = linkM.copy().astype(np.float16)
    M1_domains_to_ips = sparse.csr_matrix(M1_domains_to_ips)
    M1_domains_to_ips = M1_domains_to_ips.multiply(rep_sent_domains)
    rep_received_ips = sparse.csr_matrix(rep_received_ips + M1_domains_to_ips.sum(axis=0).reshape(D_ips,1))
    del M1_domains_to_ips
    
    M1_ips_to_domains = linkM.transpose().copy().astype(np.float16)
    M1_ips_to_domains = sparse.csr_matrix(M1_ips_to_domains)
    M1_ips_to_domains = M1_ips_to_domains.multiply(rep_sent_ips)
    rep_received_domains = sparse.csr_matrix(rep_received_domains + M1_ips_to_domains.sum(axis=0).reshape(D_domains,1))
    del M1_ips_to_domains
          
    # Update the array of reputation ex novo at each iteration
#     M1_domains_to_ips = linkM.copy() * rep_sent_domains
#     M1_ips_to_domains = linkM.transpose().copy() * rep_sent_ips
#      
#     rep_received_domains = M1_ips_to_domains.sum(axis=0).reshape(D_domains,1)
#     rep_received_ips = M1_domains_to_ips.sum(axis=0).reshape(D_ips,1)
        
    # Normalize reputation
    previous_amount_of_reputation = (np.abs(rep_sent_domains)).sum() + (np.abs(rep_sent_ips)).sum()
    current_amount_of_reputation = (np.abs(rep_received_domains)).sum() + (np.abs(rep_received_ips)).sum()
    ratio_of_reputation = float(previous_amount_of_reputation) / current_amount_of_reputation
    rep_received_domains = rep_received_domains * ratio_of_reputation * alpha_dump
    rep_received_ips = rep_received_ips * ratio_of_reputation * alpha_dump

    # Increase iter count
    iteration = iteration+1

    # Compute change
    change = (np.abs(rep_sent_domains-rep_received_domains)).sum() + (np.abs(rep_sent_ips-rep_received_ips)).sum()
    changes.append(change)
    
    # Save reputations
    writefile(rep_received_domains.toarray(), 'iter'+str(iteration)+'-reputation-domains')
    writefile(rep_received_ips.toarray(), 'iter'+str(iteration)+'-reputation-ips')
    writefile(change, 'iter'+str(iteration)+'-change')
       
    # Set the new vectors
    rep_sent_domains = sparse.csc_matrix(rep_received_domains.copy())
    rep_sent_ips = sparse.csc_matrix(rep_received_ips.copy())
    
    # Printout
    print('{0} - Iteration {1}: degree of change={2}'.format(time.strftime('%X %x'),iteration,change))
        
print( '{0} - PageRank terminated!'.format(time.strftime('%X %x')))
print( 'Final reputations of domains: {0}'.format(rep_received_domains.toarray()))
print( 'Final reputations of ips: {0}'.format(rep_received_ips.toarray()))

    
writefile(changes,'allchanges')

print('{0} - Final logs written'.format(time.strftime('%X %x')))

print('{0} - All done!'.format(time.strftime('%X %x')))

    






