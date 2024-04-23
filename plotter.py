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

    stop = time.time()

    print(">>> Dictionaries ready; elapsed time: {:.2f} seconds".format(stop-start))

    Events_0LCPlots(lc_dict, lcpf_dict, legend=["LayerCluster", "PFClusters"], c=["red", "black"],
                    save=configuration["SaveDirectory"], caloP=cp_dict, recHits=pfRec_dict)

    ClusterPlots(lc_dict, lcpf_dict, legend=["LayerCluster", "PFClusters"], c=["red", "black"],
                    save=configuration["SaveDirectory"], debug=cp_dict, recHits=pfRec_dict)
    
    exit(0)
    
    if (doPFComparison):
        EfficiencyPlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict],\
                        legend=["CLUE", "PFClustering"],
                        annotate=configuration["Annotate"],
                        save=configuration["SaveDirectory"])
        PurityPlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict],\
                    legend=["CLUE", "PFClustering"],
                    annotate=configuration["Annotate"],
                    save=configuration["SaveDirectory"])
        FakeRatePlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict],\
                    legend=["CLUE", "PFClustering"],
                    annotate=configuration["Annotate"], 
                    save=configuration["SaveDirectory"])
    if False:
        
        EfficiencyPlots([lc_dict], [cp_dict], c=['red'], legend=["CLUE"], save=configuration["SaveDirectory"])
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

