import mplhep as hep
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
import awkward as ak

from data.dictionaries import fillLCDictionary, fillCPDictionary, fillSCDictionary, pfHitsDictionary
from plotting.EfficiencyPlots import EfficiencyPlots
from plotting.PurityPlots import PurityPlots
from plotting.FakeRatePlots import FakeRatePlots
from plotting.ResponsePlot import ResponsePlot
from plotting.PUContaminationPlot import PUContaminationPlot
from plotting.ClusterPlots import ClusterPlots
from plotting.events_0LCPlots import Events_0LCPlots
from plotting.EfficiencyPlotsComparison import EfficiencyPlots_comparison

import numpy as np

from config.configuration import configuration


if __name__ == "__main__":

    dir_string = configuration["FilesDirectory"] 
    files_vec = []
    files = list(os.listdir(dir_string))
    files = [dir_string+f for f in files]
    files_vec.extend(files)

    for fileName in files_vec:
        if fileName.endswith('.root'):
            continue
        files_vec.remove(fileName)
   
    if (configuration["DebugMode"]):
        print(">>> Running in DebugMode")
        files_vec = files_vec[:10] 
    print(">>> Found %s files" %len(files_vec))
    
    #files_vec = [files_vec[2]]
    print(files_vec)

    doPFComparison = configuration["doPFComparison"]
    
    start = time.time()
    lc_dict = fillLCDictionary(files_vec, concatenate=configuration["concatenate"])
    cp_dict = fillCPDictionary(files_vec, concatenate=configuration["concatenate"])
    sc_dict = fillSCDictionary(files_vec, concatenate=configuration["concatenate"])
    pfRec_dict = pfHitsDictionary(files_vec, concatenate=configuration["concatenate"])

    if (doPFComparison):
        lcpf_dict = fillLCDictionary(files_vec, isPF=True, concatenate=configuration["concatenate"])
        cppf_dict = fillCPDictionary(files_vec, isPF=True, concatenate=configuration["concatenate"])
        scpf_dict = fillSCDictionary(files_vec, isPF=True, concatenate=configuration["concatenate"])
        
    for i in range(len(lc_dict)):
        mask_eventsWithCaloP = (ak.num(cp_dict[i]['caloParticleEta'], axis=1) > 0)[:,np.newaxis] # mask of each event
        # mask_eventsWithCaloP = [[T], [T], ..., [F], .., [T]]
        maskperEvent_eventsWithCaloP = ak.any(mask_eventsWithCaloP, axis=1) # remove events with no CaloP

        lc_dict[i] = lc_dict[i][maskperEvent_eventsWithCaloP]
        cp_dict[i] = cp_dict[i][maskperEvent_eventsWithCaloP]
        sc_dict[i] = sc_dict[i][maskperEvent_eventsWithCaloP]
        pfRec_dict[i] = pfRec_dict[i][maskperEvent_eventsWithCaloP]
        lcpf_dict[i] = lcpf_dict[i][maskperEvent_eventsWithCaloP]
        cppf_dict[i] = cppf_dict[i][maskperEvent_eventsWithCaloP]
        scpf_dict[i] = scpf_dict[i][maskperEvent_eventsWithCaloP]

    stop = time.time()

    print(">>> Dictionaries ready; elapsed time: {:.2f} seconds".format(stop-start))
    
    print("Create plots for Events with 0 LC")
    #Events_0LCPlots(lc_dict, lcpf_dict, legend=["LayerCluster", "PFClusters"], c=["red", "black"],
    #                save=configuration["SaveDirectory"], caloP=cp_dict, recHits=pfRec_dict)

    print("Create Cluster Plots")
    #ClusterPlots(lc_dict, lcpf_dict, legend=["LayerCluster", "PFClusters"], c=["red", "black"],
    #                save=configuration["SaveDirectory"], caloP=cp_dict, recHits=pfRec_dict)    
    
    if (doPFComparison):
        print("Create Efficiency Plots with comparison")
        EfficiencyPlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict], cut_sim2Reco=[1.0, 0.2],
                        legend=["CLUE", "PFClustering"],
                        annotate=configuration["Annotate"],
                        save=configuration["SaveDirectory"])
        print("Create Efficiency Plots between different kappa")
        EfficiencyPlots_comparison(
                        [lc_dict[0], lc_dict[1], lc_dict[2], lc_dict[3], lc_dict[4], lc_dict[5], lcpf_dict[0]],
                        [cp_dict[0], cp_dict[1], cp_dict[2], cp_dict[3], cp_dict[4], cp_dict[5], cppf_dict[0]],
                        compare_kappa=[0,1,2,3,4,5],
                        cut_sim2Reco=[1.0, 0.2],
                        cut_sharedEn=[0.05,0.1,0.25,0.5,0.7],
                        annotate=configuration["Annotate"],
                        save=configuration["SaveDirectory"],
                        label_kappa=['kappa-1.0', 'kappa-1.5', 'kappa-2.0', 'kappa-2.5', 'kappa-3.0', 'kappa-3.5'])
        exit(0)
        PurityPlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict],\
                    legend=["CLUE", "PFClustering"],
                    annotate=configuration["Annotate"],
                    save=configuration["SaveDirectory"])
        FakeRatePlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict],\
                    legend=["CLUE", "PFClustering"],
                    annotate=configuration["Annotate"], 
                    save=configuration["SaveDirectory"])
    else:
        print("Create Efficiency Plots without comparison")
        EfficiencyPlots([lc_dict], [cp_dict], cut_sim2Reco=[1.0, 0.2], c=['red'], legend=["CLUE"], save=configuration["SaveDirectory"])
        exit(0)
        PurityPlots([lc_dict], [cp_dict], c=['red'], legend=["CLUE"], save=configuration["SaveDirectory"])
        FakeRatePlots([lc_dict], [cp_dict], c=['red'], legend=["CLUE"], save=configuration["SaveDirectory"])

        # make specific plots
        ResponsePlot([cp_dict, cppf_dict], 
                    annotate=configuration["Annotate"],
                    legend=["CLUE", "PFClustering"],
                    c=['red', 'black'],
                    save=configuration["SaveDirectory"])

        PUContaminationPlot([lc_dict], 
                            annotate=configuration["Annotate"],
                            legend=["CLUE"],
                            c=['red'],
                            save=configuration["SaveDirectory"])


    print(">>> Plots saved in %s" % configuration["SaveDirectory"])

