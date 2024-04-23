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

    events_dictionary = {}
    labels_array = ['layerClusterEnergy', 'layerClusterEta', 'layerClusterPhi', 'recoToSimAssociation',
                    'AssociatedCP', 'layerClusterLayer', 'layerClusterPUContribution', 'layerClusterNumberOfHits']

    events_idx = 0
    pf = ""
    if isPF:
        pf = "PF"
    print(">>> Preparing %sLCDictionaries" % pf)
    for i, root_file in enumerate(tqdm(root_files_vec)):
        f = uproot.open(root_file)
        tree = f[tree_string]
        if i == 0:
            for label in labels_array:
                events_dictionary[label] = tree[label].array()
        else:
            for label in labels_array:
                events_dictionary[label] = ak.concatenate(events_dictionary[label], tree[label].array(), axis=0)

        if concatenate == False:
            break

    return events_dictionary

def fillCPDictionary(root_files_vec, isPF=False, concatenate=False):
    tree_string = "lcDumper/caloparticles"
    if (isPF):
        tree_string = "lcDumperPF/caloparticles"

    events_dictionary = {}
    labels_array = ['caloParticleEnergy', 'caloParticleEta', 'caloParticlePhi', 'simToRecoAssociation',
                    'AssociatedLC', 'sharedEnergy']
    pf = ""
    if isPF:
        pf = "PF"
    print(">>> Preparing %sCPDictionaries" % pf)
    for i, root_file in enumerate(tqdm(root_files_vec)):
        f = uproot.open(root_file)
        tree = f[tree_string]
        if i == 0:
            for label in labels_array:
                events_dictionary[label] = tree[label].array()
        else:
            for label in labels_array:
                events_dictionary[label] = ak.concatenate(events_dictionary[label], tree[label].array(), axis=0)
                
        if concatenate == False:
            break
                
    return events_dictionary

def fillSCDictionary(root_files_vec, isPF=False, concatenate=False):
    tree_string = "lcDumper/simclusters"
    if (isPF):
        tree_string = "lcDumperPF/simclusters"

    events_dictionary = {}
    labels_array = ['simClusterEnergy', 'simClusterEta', 'simClusterPhi', 'simToRecoAssociation',
                    'AssociatedLC', 'simClusterLayer', 'sharedEnergy']
    pf = ""
    if isPF:
        pf = "PF"
    #print(">>> Preparing %sSCDictionaries" % s)
    for i, root_file in enumerate(tqdm(root_files_vec)):
        f = uproot.open(root_file)
        tree = f[tree_string]
        
        if i == 0:
            for label in labels_array:
                events_dictionary[label] = tree[label].array()
        else:
            for label in labels_array:
                events_dictionary[label] = ak.concatenate(events_dictionary[label], tree[label].array(), axis=0)
                
        if concatenate == False:
            break
                
    return events_dictionary
