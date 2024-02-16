from utils.fillVectors import fillVector, fillVectorOfVectors
from config.configuration import configuration
import uproot
from numba import njit

def fillLCDictionary(root_files_vec, isPF=False):
    tree_string = "lcDumper/layerclusters"
    if (isPF):
        tree_string = "lcDumperPF/layerclusters"

    events_dictionary = {}
    events_idx = 0
    for root_file in root_files_vec:
        f = uproot.open(root_file)
        tree = f[tree_string]
        for event in range(0, len(tree["layerClusterEnergy"].array())):
            events_str = str(events_idx+1)
            events_dictionary[events_str] = {}
            events_dictionary[events_str]["LayerClustersEnergy"] = fillVector(tree["layerClusterEnergy"].array()[event])
            events_dictionary[events_str]["LayerClustersEta"] = fillVector(tree["layerClusterEta"].array()[event])
            events_dictionary[events_str]["LayerClustersPhi"] = fillVector(tree["layerClusterPhi"].array()[event])
            events_dictionary[events_str]["LC2CPscore"] = fillVectorOfVectors(tree["recoToSimAssociation"].array()[event])
            events_dictionary[events_str]["AssociatedCP"] = fillVectorOfVectors(tree["AssociatedCP"].array()[event])
            events_dictionary[events_str]["LayerClustersLayer"] = fillVector(tree["layerClusterLayer"].array()[event])
            events_dictionary[events_str]["PUContamination"] = fillVector(tree["layerClusterPUContribution"].array()[event])
            events_dictionary[events_str]["LayerClustersNHits"] = fillVector(tree["layerClusterNumberOfHits"].array()[event])
            if (configuration["DebugMode"]):
                print(">>> LCDictionary: finished processing event %s" % events_idx)
            events_idx += 1
    return events_dictionary

def fillCPDictionary(root_files_vec, isPF=False):
    tree_string = "lcDumper/caloparticles"
    if (isPF):
        tree_string = "lcDumperPF/caloparticles"

    events_dictionary = {}
    events_idx = 0
    for root_file in root_files_vec:
        f = uproot.open(root_file)
        tree = f[tree_string]
        for event in range(0, len(tree["caloParticleEnergy"].array())):
            events_str = str(events_idx+1)
            events_dictionary[events_str] = {}
            events_dictionary[events_str]["CaloParticleEnergy"] = fillVector(tree["caloParticleEnergy"].array()[event])
            events_dictionary[events_str]["CaloParticleEta"] = fillVector(tree["caloParticleEta"].array()[event])
            events_dictionary[events_str]["CaloParticlePhi"] = fillVector(tree["caloParticlePhi"].array()[event])
            events_dictionary[events_str]["CP2LCscore"] = fillVectorOfVectors(tree["simToRecoAssociation"].array()[event])
            events_dictionary[events_str]["AssociatedLC"] = fillVectorOfVectors(tree["AssociatedLC"].array()[event])
            events_dictionary[events_str]["SharedEnergy"] = fillVectorOfVectors(tree["sharedEnergy"].array()[event])
            #events_dictionary[events_str]["BunchCrossing"] = fillVector(tree["caloParticleBX"].array()[event])
            if (configuration["DebugMode"]):
                print(">>> CPDictionary: finished processing event %s" % events_idx)
            events_idx += 1
    return events_dictionary

def fillSCDictionary(root_files_vec, isPF=False):
    tree_string = "lcDumper/simclusters"
    if (isPF):
        tree_string = "lcDumperPF/simclusters"

    events_dictionary = {}
    events_idx = 0
    for root_file in root_files_vec:
        f = uproot.open(root_file)
        tree = f[tree_string]
        for event in range(0, len(tree["simClusterEnergy"].array())):
            events_str = str(events_idx+1)
            events_dictionary[events_str] = {}
            events_dictionary[events_str]["SimClusterEnergy"] = fillVector(tree["simClusterEnergy"].array()[event])
            events_dictionary[events_str]["SimClusterEta"] = fillVector(tree["simClusterEta"].array()[event])
            events_dictionary[events_str]["SimClusterPhi"] = fillVector(tree["simClusterPhi"].array()[event])
            events_dictionary[events_str]["SC2LCscore"] = fillVectorOfVectors(tree["simToRecoAssociation"].array()[event])
            events_dictionary[events_str]["AssociatedLC"] = fillVectorOfVectors(tree["AssociatedLC"].array()[event])
            events_dictionary[events_str]["SimClusterLayer"] = fillVector(tree["simClusterLayer"].array()[event])
            events_dictionary[events_str]["SharedEnergy"] = fillVectorOfVectors(tree["sharedEnergy"].array()[event])
            if (configuration["DebugMode"]):
                print(">>> SCDictionary: finished processing event %s" % events_idx)
            events_idx += 1
    return events_dictionary
