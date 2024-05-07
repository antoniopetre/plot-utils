from utils.fillVectors import fillVector, fillVectorOfVectors
from config.configuration import configuration
import uproot
from numba import njit
from tqdm import tqdm
import awkward as ak

# TODO: concat doesn't work
def fillLCDictionary(root_files_vec, isPF=False, concatenate=False):
    tree_string = "lcDumper/layerclusters"
    if (isPF):
        tree_string = "lcDumperPF/layerclusters"

    labels_array = ['layerClusterEnergy', 'layerClusterEta', 'layerClusterPhi', 'recoToSimAssociation',
                    'AssociatedCP', 'layerClusterLayer', 'layerClusterPUContribution', 'layerClusterNumberOfHits']
    
    dictionary = {}
    for i in labels_array:
        dictionary[i] = []
            
    events_array = []

    events_idx = 0
    pf = ""
    if isPF:
        pf = "PF"
    print(">>> Preparing %sLCDictionaries" % pf)
    for i, root_file in enumerate(tqdm(root_files_vec)):
        
        events_array_temporal = ak.Array([dictionary])
        f = uproot.open(root_file)
        tree = f[tree_string]
        for label in labels_array:
            events_array_temporal[label] = tree[label].array()
        
        events_array.append(events_array_temporal)

        if concatenate == False:
            break

    return events_array

def fillCPDictionary(root_files_vec, isPF=False, concatenate=False):
    tree_string = "lcDumper/caloparticles"
    if (isPF):
        tree_string = "lcDumperPF/caloparticles"

    labels_array = ['caloParticleEnergy', 'caloParticleEta', 'caloParticlePhi', 'simToRecoAssociation',
                    'AssociatedLC', 'sharedEnergy']
    dictionary = {}
    for i in labels_array:
        dictionary[i] = []
        
    events_array = []
    
    pf = ""
    if isPF:
        pf = "PF"
    print(">>> Preparing %sCPDictionaries" % pf)
    for i, root_file in enumerate(tqdm(root_files_vec)):
        
        events_array_temporal = ak.Array([dictionary])
        f = uproot.open(root_file)
        tree = f[tree_string]
        for label in labels_array:
            events_array_temporal[label] = tree[label].array()
            
        events_array.append(events_array_temporal)
       
        if concatenate == False:
            break
                
    return events_array

def fillSCDictionary(root_files_vec, isPF=False, concatenate=False):
    tree_string = "lcDumper/simclusters"
    if (isPF):
        tree_string = "lcDumperPF/simclusters"

    labels_array = ['simClusterEnergy', 'simClusterEta', 'simClusterPhi', 'simToRecoAssociation',
                    'AssociatedLC', 'simClusterLayer', 'sharedEnergy']
    
    dictionary = {}
    for i in labels_array:
        dictionary[i] = []
        
    events_array = []
    
    pf = ""
    if isPF:
        pf = "PF"
    #print(">>> Preparing %sSCDictionaries" % s)
    for i, root_file in enumerate(tqdm(root_files_vec)):
        
        events_array_temporal = ak.Array([dictionary])
        f = uproot.open(root_file)
        tree = f[tree_string]
        
        for label in labels_array:
            events_array_temporal[label] = tree[label].array()
       
        events_array.append(events_array_temporal)        
        
        if concatenate == False:
            break
                
    return events_array


def pfHitsDictionary(root_files_vec, isPF=False, concatenate=False):
    tree_string = "lcDumper/pfrechits"
    if (isPF):
        tree_string = "lcDumperPF/pfrechits"

    labels_array = ['pfrechitEta', 'pfrechitPhi', 'pfrechitEnergy', 'pfrechitTime']
    
    dictionary = {}
    for i in labels_array:
        dictionary[i] = []
        
    events_array = []
    
    pf = ""
    if isPF:
        pf = "PF"
    #print(">>> Preparing %sSCDictionaries" % s)
    for i, root_file in enumerate(tqdm(root_files_vec)):
        
        events_array_temporal = ak.Array([dictionary])
        f = uproot.open(root_file)
        tree = f[tree_string]
        
        for label in labels_array:
            events_array_temporal[label] = tree[label].array()
            
        events_array.append(events_array_temporal)
                
        if concatenate == False:
            break
                
    return events_array
